# common.py
import math
import logging
import inspect
import torch
import re
import torch.nn.functional as F
import comfy.sample
import comfy.utils
import comfy.model_management
import comfy.patcher_extension
import comfy.ldm.common_dit
import node_helpers
from einops import rearrange
from comfy.ldm.flux.layers import timestep_embedding

LATENT_SCALE = 16
_CONTEXT_EDGE = 0.35
_EDGE_FALLOFF = 0.075
_AUTO_TILE_MIN, _AUTO_TILE_MAX = 1024, 1536
_DETAIL_SHARPNESS = 5
_DETAIL_RADIUS = 0.35


def _pixel_alignment(model):
    """
    How many pixels one model token covers: the VAE's spatial compression times the DiT's patch
    size, read from the loaded model so any model family lands on the right grid automatically.
    """
    try:
        downscale = int(model.get_model_object("latent_format").spacial_downscale_ratio)
    except Exception:
        logging.warning("[TiledUpscale] could not read the model's latent format; assuming %dpx tokens.", LATENT_SCALE)
        return LATENT_SCALE
    patch = 1
    try:
        diffusion_model = model.get_model_object("diffusion_model")
        for attr in ("patch_size", "patch"):
            value = getattr(diffusion_model, attr, None)
            if isinstance(value, (tuple, list)):
                value = max(value) if value else None
            if isinstance(value, int) and value > 0:
                patch = value
                break
    except Exception:
        pass
    return max(1, downscale * patch)


def _reference_method(model, conditioning):
    """
    The method name to attach to the conditioning so the model actually reads its reference
    latents, or None to leave the conditioning as it is. Only models that declare
    `default_ref_method = None` (Krea 2) need "index_timestep_zero" filled in; every other model
    either has its own default (FLUX.2/Qwen "index", FLUX.1 "offset") or ignores
    reference_latents entirely (plain t2i models), which is harmless.
    """
    if any("reference_latents_method" in entry[1] for entry in conditioning):
        return None  
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception:
        return None
    dm_name = type(diffusion_model).__name__.lower()
    if "flux" in dm_name and "flux3" not in dm_name:
        return "index"
    if getattr(diffusion_model, "default_ref_method", "unset") is None:
        return "index_timestep_zero"
    return None


def _tokenizer_handles_images(clip):
    """按 tokenize_with_weights 的签名判断，而不是猜类名。klein 的
    KleinTokenizer 签名是 (text, return_word_ids, llama_template, **kwargs)，
    没有 images 形参；QwenImageTokenizer / Qwen Image 2.1 的签名里真有 images。"""
    try:
        tok = clip.tokenizer
    except Exception:
        return False

    def _accepts_images(fn):
        try:
            return "images" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    candidates = []
    for value in vars(tok).values():
        if isinstance(value, dict):
            candidates.extend(value.values())
        else:
            candidates.append(value)

    fn = getattr(tok, "tokenize_with_weights", None)
    if fn is not None and _accepts_images(fn):
        return True
    for sub in candidates:
        sub_fn = getattr(sub, "tokenize_with_weights", None)
        if sub_fn is not None and _accepts_images(sub_fn):
            return True

    names = {type(tok).__name__.lower()} | {type(s).__name__.lower() for s in candidates}
    return any(("vl" in n) or ("krea" in n) or ("qwen_image" in n) or ("qwenimage" in n) for n in names)

def _is_flux2_family(model):
    """FLUX.2/klein 系 DiT（类名含 flux）。它的 conditioning 训练分布里没有
    vision 占位符、没有 Qwen-VL 的 describe 系统提示；per-tile 重编码只会让
    每个 tile 拿到分布外条件，表现为部分区域脏/重影。永远走 reference-latent 路径。"""
    try:
        dm = model.get_model_object("diffusion_model")
        return "flux" in type(dm).__name__.lower()
    except Exception:
        return False

# ---------------------------------------------------------------------------
# reference_strength 每步调度
# 语义：强度列表与 steps 一一对应。第 i 次模型前向（第 i 个采样步，0 起）取
# 第 i 个值；列表比步数短则重复末值，比步数长则只用前段。强度只由步序号决定
# —— timesteps 仅用于识别"下一步 / 新一轮 / 同一步"，数值不参与强度计算，
# 因此 denoise 取任何值（只改 sigma 范围，不改前向次数）对本调度完全透明。
# 作用方式（guidance）：pred = pred_无参考 + s * (pred_有参考 - pred_无参考)。
# 两路前向都在训练分布内，不向参考注入任何噪声；s=1 等于原行为，s=0 该步
# 完全无参考。全 1 列表零开销；纯 0/1 走 binary（摘除参考，单倍算力）。
# ---------------------------------------------------------------------------
_STRENGTH_SPLIT = re.compile(r"[,，、:：;；\s]+")

def _parse_strength_schedule(value, default="1.0"):
    """'1.0，0.9、0.7：0.5' -> [1.0, 0.9, 0.7, 0.5]（中英文逗号/顿号/冒号均可）。"""
    if value is None:
        value = default
    if isinstance(value, (int, float)):
        text = str(value)                      
    else:
        text = str(value).strip()
    if not text:
        text = default
    parts = [p for p in _STRENGTH_SPLIT.split(text) if p]
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(
            f"reference_strength 无法解析: {value!r}（小数请用英文句点，如 0.8）") from e
    out = []
    for v in values:
        if not (0.0 <= v <= 1.0):
            logging.warning("[TileSchedule] reference_strength=%s 超出 [0,1]，已钳制", v)
            v = min(1.0, max(0.0, v))
        out.append(v)
    return out if out else [1.0]


class _StepSchedule:
    """按步序号查表。一次模型前向 = 一步：
    - timestep 下降   -> 下一步（step+1）
    - timestep 反弹   -> 新一轮采样（新 tile / redo 首块），step 归零
    - timestep 相等   -> 同一步的另一路前向（cond/uncond 或合批），不推进
    """

    def __init__(self, values):
        self.values = values
        self.last_t = None
        self.step = 0

    def strength(self, timesteps):
        t = float(timesteps.reshape(-1)[0]) if torch.is_tensor(timesteps) else float(timesteps)
        if self.last_t is not None:
            if t > self.last_t + 1e-6:          
                self.step = 0
            elif t < self.last_t - 1e-6:        
                self.step += 1
        self.last_t = t
        return self.values[min(self.step, len(self.values) - 1)]


def _refs_are_present(refs):
    """判断本步是否带有参考。Krea2-conditioning 链里 ref_latents 由 Patch 的
    extra_conds 以 CONDList 注入，到达 wrapper 时可能尚未解包成张量列表，
    因此除 None 之外一律按"有参考"处理，避免误判为无参考而透传。"""
    if refs is None:
        return False
    if torch.is_tensor(refs):
        return True
    if isinstance(refs, (list, tuple)):
        return len(refs) > 0
    return True    


def _find_ref_slot(args, kwargs):
    """DiT forward 各模型族的 ref 传递位置不同：Qwen/FLUX.2/FLUX.3(蒸馏慢路径) 把
    ref_latents 按位置参数传进 wrapper，且是 list 形态（args 里的某个位置）；
    FLUX.3 base 在 inference_mode 快路径下改发 kwargs 里的 ref_states/ref_shapes。
    返回 ("args", i) / ("kwargs", key) / ("flux3", None) / (None, None)。"""
    for key in ("ref_latents", "reference_latents"):
        if kwargs.get(key) is not None:
            return ("kwargs", key)
    if kwargs.get("ref_states") is not None:
        return ("flux3", None)
    for i, a in enumerate(args):
        if torch.is_tensor(a) and a.ndim == 5:
            return ("args", i)
        if isinstance(a, (list, tuple)) and len(a) > 0 and torch.is_tensor(a[0]):
            return ("args", i)
    return (None, None)


def _set_refs(args, kwargs, kind, slot, value):
    """返回把 ref 槽位替换成 value（None = 摘除参考）后的 (args, kwargs)。"""
    if kind == "flux3":
        # FLUX.3 快路径：参考只能整体摘除（value 忽略，此路径不做换参考）。
        kwargs = dict(kwargs)
        kwargs["ref_states"] = None
        kwargs["ref_shapes"] = None
        return args, kwargs
    if kind == "args":
        return args[:slot] + (value,) + args[slot + 1:], kwargs
    kwargs = dict(kwargs)
    kwargs[slot] = value
    return args, kwargs

# ---------------------------------------------------------------------------
# 上游 prefix-cache bug 探测（Qwen Image 2.1，Comfy-Org/ComfyUI #16511/#16947）
# comfy/ldm/wan/model_animate2.py 的 PoseBranchCache.select 用 self.slots.remove(s)
# 做 MRU 移动：当目标槽前面躺着形状不同的另一个槽时（有参考/无参考两种组成
# 交替就会出现），dict 比较会对 key 张量执行 ==，直接 RuntimeError。
# 上游修复该行后行为消失。结果进程内只判定一次，ComfyUI 更新后重启自动重判。
# ---------------------------------------------------------------------------
_NOREF_CACHE_BYPASS = None

def _probe_pose_branch_cache(cls):
    """对已安装核心的 PoseBranchCache.select 做真实行为探测。
    返回 True=仍有 bug / False=已修复 / None=探针无法进行（结构变了）。
    生产场景里 K_noref 槽排在 K_ref 槽后面时 list.remove 崩，探针原样复现。"""
    c = object.__new__(cls)            # 绕过 __init__，手工布置最小状态
    c._pending = {}                    # select 开头会检查它
    c.slot = None
    c.slots = [{"key": torch.zeros(1, 8)},   # 前槽：形状不同（等价 K_ref 槽）
               {"key": torch.zeros(1, 4)}]   # 目标槽（等价 K_noref 槽）
    try:
        hit = cls.select(c, torch.zeros(1, 4), create=False)
    except RuntimeError:
        return True                     # 跨形状比较崩溃 = 未修复
    except Exception:
        return None                     # 探针失效，交回退逻辑
    return bool(hit) and c.slot is c.slots[-1]

def _noref_cache_bypass_needed():
    """True = 摘除参考的前向需要隔离 prefix cache。进程内只判定一次。"""
    global _NOREF_CACHE_BYPASS
    if _NOREF_CACHE_BYPASS is not None:
        return _NOREF_CACHE_BYPASS
    result = True   # 判定不了时按"有问题"处理：bypass 只是少复用，不会错
    try:
        import comfy.ldm.wan.model_animate2 as _m2
        cls = getattr(_m2, "PoseBranchCache", None)
        if cls is not None and hasattr(cls, "select"):
            probe = _probe_pose_branch_cache(cls)
            if probe is not None:
                result = probe
            else:
                # 回退：源码扫描。修复版不会保留这行。
                result = "self.slots.remove(" in inspect.getsource(cls.select)
    except Exception:
        pass
    if result:
        logging.info("[TileSchedule] core PoseBranchCache.select still has the "
                     "list.remove bug (#16511/#16947); reference-stripped forwards "
                     "will bypass the Qwen Image 2.1 prefix cache.")
    else:
        logging.info("[TileSchedule] core prefix cache looks fixed; normal logic.")
    _NOREF_CACHE_BYPASS = result
    return result

def _uses_prefix_cache(dm):
    """只有 Qwen Image 2.1 的 transformer 挂着这套缓存（select_prefix_cache 是它的方法）。
    FLUX.2 / FLUX.3 / Qwen 2509/2511 在此直接放行，零开销。"""
    return dm is not None and hasattr(dm, "select_prefix_cache")

def _find_topts(args, kwargs):
    """transformer_options 的位置：qwen2.1 的 forward 把它按位置参数传进 wrapper
    （execute(x, t, ctx, ref_latents, image_slots, transformer_options)），也可能在 kwargs。"""
    if isinstance(kwargs.get("transformer_options"), dict):
        return ("kwargs", "transformer_options")
    for i in range(len(args) - 1, -1, -1):
        if isinstance(args[i], dict):
            return ("args", i)
    return (None, None)

def _drop_refs_no_cache(executor, args, kwargs):
    """摘除参考的那次前向：浅拷 transformer_options，关掉 prefix cache。
    依据核心文档化开关：select_prefix_cache 见 options['device']=='off'
    直接返回 (None, False) —— 该次前向既不读也不写缓存，有参考主路缓存不受影响。"""
    if not _uses_prefix_cache(getattr(executor, "class_obj", None)):
        return args, kwargs
    kind, slot = _find_topts(args, kwargs)
    if kind == "kwargs":
        opts = dict(kwargs["transformer_options"])
        opts["qwen_image21_cache"] = {"device": "off"}
        kwargs = dict(kwargs); kwargs["transformer_options"] = opts
    elif kind == "args":
        opts = dict(args[slot])
        opts["qwen_image21_cache"] = {"device": "off"}
        args = args[:slot] + (opts,) + args[slot + 1:]
    return args, kwargs



def _scheduled_ref_model(model, strength_schedule):
    """reference-latent 路径（Qwen Image 2.1 / Edit 2509/2511 / FLUX.2 等）的每步
    参考强度调度：DIFFUSION_MODEL wrapper 在每次前向按当前步强度做预测差值引导：
        pred = pred_无参考 + s * (pred_有参考 - pred_无参考)
    两路前向都在训练分布内（有/无参考都是这些统一模型训练过的状态），不向参考
    注入任何噪声。s=1 与原行为完全一致；s=0 该步完全无参考。
    全 1 列表 -> 原样返回零开销；只含 0/1 -> binary 模式（直接摘除参考，单倍算力）；
    含小数 -> guidance 模式（每步双前向，≈cfg=2 成本）。
    注意 x 兼容两种形态：4D (B,C,H,W)（Qwen 2.1 / FLUX.2 klein 等图像 latent）和
    5D (B,C,T,H,W)（Qwen Edit 2509/2511 用的 Wan21 视频式 latent，T=1）——之前只
    放行 4D，导致 2509/2511 在这里被静默跳过、调度完全失效。
    """
    values = strength_schedule or [1.0]
    if all(v >= 0.999 for v in values):
        return model
    binary = all(v <= 0.001 or v >= 0.999 for v in values)
    schedule = _StepSchedule(values)
    patched = model.clone()
    state = {"flux3_logged": False}
    safe_strip = _noref_cache_bypass_needed()   # 核心有 bug 才隔离缓存

    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        kind, slot = _find_ref_slot(args, kwargs)
        # QwenImageTransformer2DModel 把 ref_latents 按位置参数传入
        # (.execute(x, t, ctx, attn_mask, ref_latents, ...)，位于 args[1])，
        # _find_ref_slot 的 list/5D 张量分支均可命中；5D x 同样合法。

        #------------debug-------------
        # if not state.get("slot_logged"):
            # state["slot_logged"] = True
            # logging.info("[TileSchedule][debug] ref slot detected: kind=%s slot=%s",
                         # kind, slot)
        #-------------------------------
        
        if kind is None or x.ndim not in (4, 5):
            return executor(x, timesteps, context, *args, **kwargs)
        if kind == "flux3":
            refs = kwargs["ref_states"]   
            if not state["flux3_logged"]:
                state["flux3_logged"] = True
                logging.info("[TileSchedule] FLUX.3 inference-mode fast path engaged "
                             "(refs arrive as preprocessed ref_states); per-step "
                             "reference strength control active.")
        else:
            refs = args[slot] if kind == "args" else kwargs[slot]
        if not _refs_are_present(refs):
            return executor(x, timesteps, context, *args, **kwargs)
        s = schedule.strength(timesteps)
        if s >= 0.999:
            return executor(x, timesteps, context, *args, **kwargs)
        if binary or s <= 0.001:
            args2, kwargs2 = _set_refs(args, kwargs, kind, slot, None)
            if safe_strip:                                       
                args2, kwargs2 = _drop_refs_no_cache(executor, args2, kwargs2)
            return executor(x, timesteps, context, *args2, **kwargs2)

        # guidance：双前向 + 预测差值外推
        args2, kwargs2 = _set_refs(args, kwargs, kind, slot, None)
        out_with = executor(x, timesteps, context, *args, **kwargs)
        if safe_strip:                                            
            args2, kwargs2 = _drop_refs_no_cache(executor, args2, kwargs2)
        out_without = executor(x, timesteps, context, *args2, **kwargs2)
        return out_without + s * (out_with - out_without)


    options = patched.model_options.setdefault("transformer_options", {})
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "tiled_upscale_strength_schedule", wrapper, options)
    _promote_wrapper_key(options, "tiled_upscale_strength_schedule")
    return patched


# ---------------------------------------------------------------------------
# 上游自产 patch 的摘除（防叠加）
# 若上游已挂过（如 loader -> ElementStrengthSchedule -> TiledUpscale），
# 同一参考会被处理两次。摘除后重挂 = 后挂者生效。
# ---------------------------------------------------------------------------
_ELEMENT_PATCH_KEYS = ("tiled_upscale_strength_schedule",
                       "tiled_upscale_in_context",
                       "tiled_upscale_in_context_cond")

def _strip_owned_patches(model, keys=_ELEMENT_PATCH_KEYS, strip_extra_conds=True):
    """clone 模型并摘除 keys 指定的 wrapper，以及本扩展标记过的 extra_conds
    object patch。core 的 ModelPatcher.clone() 会深拷贝 model_options、浅拷贝
    object_patches，因此删除只作用于克隆体，上游 ModelPatcher 不受影响。"""
    patched = model.clone()
    removed = []
    topts = patched.model_options.get("transformer_options")
    if isinstance(topts, dict):
        wrappers = topts.get("wrappers")
        if isinstance(wrappers, dict):
            dm = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL)
            if isinstance(dm, dict):
                for key in keys:
                    if key in dm:
                        del dm[key]
                        removed.append(key)
    if strip_extra_conds:
        fn = patched.object_patches.get("extra_conds")
        if fn is not None and getattr(fn, "_element_krea2_extra_conds", False):
            del patched.object_patches["extra_conds"]
            removed.append("extra_conds(krea2)")
    if removed:
        logging.info("[Element] removed upstream patches %s before applying own logic.", removed)
    return patched

def _promote_wrapper_key(options, key):
    """把 key 指定的 DIFFUSION_MODEL wrapper 移到执行链最外层。
    core 事实（patcher_extension.py）：get_all_wrappers 按 wrapper key 的插入
    顺序展平成列表，WrapperExecutor 从列表首位开始执行（首位 = 最外层）。
    上游 Krea2 In-Context Patch 的 wrapper 在参考存在时会短路内层（不调用
    executor、直接手动跑 DiT），调度 wrapper 必须位于其外侧才能先摘下参考
    槽位做逐步引导，否则调度会静默失效。"""
    wrappers = options.get("wrappers")
    if not isinstance(wrappers, dict):
        return
    dm = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL)
    if not isinstance(dm, dict) or key not in dm:
        return
    if next(iter(dm)) == key:          
        return
    ours = dm.pop(key)
    reordered = {key: ours}
    reordered.update(dm)
    wrappers[comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL] = reordered


# ---------------------------------------------------------------------------
# Krea 2 identity-edit in-context path. Layout follows ComfyUI-Krea2Edit
# ---------------------------------------------------------------------------

_KREA2_PARTS = ("patch", "channels", "tdim", "first", "last", "blocks", "tmlp", "tproj",
                "txtfusion", "txtmlp", "pe_embedder", "_unpack_context")


def _is_krea2(diffusion_model):
    return all(hasattr(diffusion_model, part) for part in _KREA2_PARTS)


def _img_ids(batch, frame, h, w, device):
    """3-axis RoPE ids for one image block: axis 0 is the frame index, axes 1-2 the grid."""
    ids = torch.zeros(h, w, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None]
    ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :]
    return ids.reshape(1, h * w, 3).repeat(batch, 1, 1)


def _to_4d(latent):
    """(B,C,T,H,W) -> (B*T,C,H,W); 4D passes through. Images are T=1."""
    if latent.ndim == 5:
        b, c, t, h, w = latent.shape
        return latent.reshape(b * t, c, h, w)
    return latent


def _krea2_in_context_forward(dm, x, timesteps, context, source, transformer_options):
    """Krea 2 identity-edit 前向：[text | source(frame=1) | target(frame=0)]。
    source=None 时构建原生 T2I 布局 [text | target(frame=0)]（= base 模型的
    训练前向），供 guidance 模式的无参考路使用。任何情况下不修改 source 数值。"""
    patch = dm.patch
    temporal = x.ndim == 5
    if temporal:
        b5, _, t5, _, _ = x.shape
        x = _to_4d(x)
    batch, _, h_orig, w_orig = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    height, width = x.shape[-2:]
    h_, w_ = height // patch, width // patch
    to_tokens = lambda t: dm.first(rearrange(t, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                                             ph=patch, pw=patch))
    target_tokens = to_tokens(x)
    if source is not None:
        src = _to_4d(source).to(device=x.device, dtype=x.dtype)
        if src.shape[0] != batch:
            src = src[:1].expand(batch, *src.shape[1:])
        if src.shape[-2:] != (height, width):
            src = F.interpolate(src.float(), size=(height, width), mode="bilinear").to(x.dtype)
        src = comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch))
        source_tokens = to_tokens(src)
    else:
        source_tokens = None
    t = dm.tmlp(timestep_embedding(timesteps, dm.tdim).unsqueeze(1).to(target_tokens.dtype))
    tvec = dm.tproj(t)
    context = dm.txtfusion(dm._unpack_context(context), mask=None, transformer_options=transformer_options)
    context = dm.txtmlp(context)
    txt_len, tgt_len = context.shape[1], target_tokens.shape[1]
    src_len = 0 if source_tokens is None else source_tokens.shape[1]
    if source_tokens is not None:
        combined = torch.cat([context, source_tokens, target_tokens], dim=1)
        pos = torch.cat([torch.zeros(batch, txt_len, 3, device=combined.device, dtype=torch.float32),
                         _img_ids(batch, 1, h_, w_, combined.device),
                         _img_ids(batch, 0, h_, w_, combined.device)], dim=1)
    else:
        combined = torch.cat([context, target_tokens], dim=1)
        pos = torch.cat([torch.zeros(batch, txt_len, 3, device=combined.device, dtype=torch.float32),
                         _img_ids(batch, 0, h_, w_, combined.device)], dim=1)
    freqs = dm.pe_embedder(pos)
    transformer_options = dict(transformer_options)
    transformer_options["total_blocks"] = len(dm.blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [txt_len + src_len, combined.shape[1]]
    for index, block in enumerate(dm.blocks):
        transformer_options["block_index"] = index
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)
    final = dm.last(combined, t)
    out = final[:, txt_len + src_len:txt_len + src_len + tgt_len, :]
    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_, w=w_, ph=patch, pw=patch, c=dm.channels)
    out = out[:, :, :h_orig, :w_orig]
    if temporal:
        out = out.reshape(b5, t5, dm.channels, h_orig, w_orig).movedim(1, 2)
    return out


def _in_context_model(model, source_holder, strength_schedule=None):
    """source_holder 驱动的 Krea 2 in-context 前向；strength_schedule 非 None 时
    按当前步强度做 guidance 差值（有 source / 无 source 两路都在分布内）。"""
    patched = model.clone()
    values = strength_schedule or [1.0]
    schedule = None if all(v >= 0.999 for v in values) else _StepSchedule(values)

    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        source = source_holder["latent"]
        if source is None:
            return executor(x, timesteps, context, *args, **kwargs)
        options = kwargs.get("transformer_options")
        if options is None:
            options = next((arg for arg in reversed(args) if isinstance(arg, dict)), {})
        dm = executor.class_obj
        if schedule is None:
            return _krea2_in_context_forward(dm, x, timesteps, context, source, options)
        s = schedule.strength(timesteps)
        if s >= 0.999:
            return _krea2_in_context_forward(dm, x, timesteps, context, source, options)
        if s <= 0.001:
            return _krea2_in_context_forward(dm, x, timesteps, context, None, options)
        out_with = _krea2_in_context_forward(dm, x, timesteps, context, source, options)
        out_without = _krea2_in_context_forward(dm, x, timesteps, context, None, options)
        return out_without + s * (out_with - out_without)

    options = patched.model_options.setdefault("transformer_options", {})
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "tiled_upscale_strength_schedule", wrapper, options)
    _promote_wrapper_key(options, "tiled_upscale_strength_schedule")
    return patched


_GROUNDING_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)


def _grounded_conditioning(clip, prompt, image, grounding_px):
    """（原 docstring 保留）vision 槽不做静态放松：reference_strength 已改为
    逐步调度，只作用于 latent 参考通道；进 vision 槽的永远是原始 tile 像素。"""
    samples = image.movedim(-1, 1)
    h, w = samples.shape[2], samples.shape[3]
    if grounding_px and max(h, w) > grounding_px:
        scale = grounding_px / max(h, w)
        samples = comfy.utils.common_upscale(samples, round(w * scale), round(h * scale), "area", "disabled")
    grounded = samples.movedim(1, -1)[:, :, :, :3]
    tokens = clip.tokenize(prompt, images=[grounded], llama_template=_GROUNDING_TEMPLATE)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return [[entry[0].to("cpu"), {k: (v.to("cpu") if torch.is_tensor(v) else v) for k, v in entry[1].items()}]
            for entry in conditioning]


def _grounded_reference_conditioning(clip, prompt, vae, tile_pixels, grounding_px,
                                     reference_method=None, encode_negative=False, qwen21=False,
                                     global_pixels=None, global_ref_latent=None, negative_prompt=""):
    samples = tile_pixels[:1].movedim(-1, 1)
    dual = global_pixels is not None and global_ref_latent is not None
    if qwen21:
        s = tile_pixels[:1]
        t_rgb = s[:, :, :, :3]
        if s.shape[-1] > 3:
            t_rgb = t_rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
        t_latent = vae.encode(s)
        if dual:
            g_rgb = global_pixels[:, :, :, :3]
            if global_pixels.shape[-1] > 3:
                g_rgb = g_rgb * global_pixels[:, :, :, 3:] + (1.0 - global_pixels[:, :, :, 3:])
            ref_latents = [t_latent, global_ref_latent]
            images = [t_rgb, g_rgb]
            full_prompt = ("Image 1 is the local region being refined. Image 2 is the complete "
                           "picture that image 1 belongs to. " + prompt)
        else:
            ref_latents = [t_latent]
            images = [t_rgb]
            full_prompt = prompt
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(full_prompt, images=images, prevent_empty_text=True))
    else:
        # 2509/2511：静态放松已移除，参考 = 原始 tile latent，削弱由前向内的调度完成
        ref_latents = [vae.encode(tile_pixels[:1])]
        total = int(384 * 384) if grounding_px <= 0 or grounding_px >= 384 else int(grounding_px * grounding_px)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = max(1, round(samples.shape[3] * scale_by))
        height = max(1, round(samples.shape[2] * scale_by))
        vl = comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)
        rgb = vl[:, :, :, :3]
        if vl.shape[-1] > 3:
            rgb = rgb * vl[:, :, :, 3:] + (1.0 - vl[:, :, :, 3:])
        images = [rgb]
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize("Picture 1: " + prompt, images=images, prevent_empty_text=True))
    extra = {"reference_latents": ref_latents}
    if reference_method is not None:
        extra["reference_latents_method"] = reference_method
    positive = node_helpers.conditioning_set_values(positive, extra, append=True)
    negative = None
    if encode_negative:
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt or "", images=images, prevent_empty_text=True))
        negative = node_helpers.conditioning_set_values(negative, extra, append=True)

    def _to_cpu_deep(v):
        if torch.is_tensor(v):
            return v.to("cpu")
        if isinstance(v, list):
            return [_to_cpu_deep(t) for t in v]
        return v
    park = lambda cond: [[e[0].to("cpu"), {k: _to_cpu_deep(v) for k, v in e[1].items()}] for e in cond]
    return park(positive), (park(negative) if negative is not None else None)


def _build_dual_ref_conditioning(clip, prompt, vae, global_pixels, tile_pixels, grounding_px,
                                 reference_method=None, encode_negative=False, qwen21=False,
                                 global_strength=0.5, tile_strength=1.0, global_ref_latent=None):
    """
    Dual-reference conditioning: a global whole-image reference (semantic anchor,
    "this tile lives inside this picture") + a local tile reference (pixel-level
    answer). Both go to the DiT as reference_latents [global, tile] AND to the
    text encoder's vision slots in the same order, matching the Qwen Edit
    2509/2511 multi-image training distribution.
    """
    # --- global reference latent: low-res, low-strength, hard-capped ---
    if global_ref_latent is None:
        g_low = _relaxed_reference_pixels(global_pixels, global_strength)
        cap = 768
        _, gh, gw, _ = g_low.shape
        if max(gh, gw) > cap:
            s = cap / max(gh, gw)
            nh = max(16, round(gh * s / 16) * 16)
            nw = max(16, round(gw * s / 16) * 16)
            gbchw = g_low.movedim(-1, 1).float()
            g_low = F.interpolate(gbchw, size=(nh, nw), mode="area").to(g_low.dtype).movedim(1, -1)
        try:
            global_ref_latent = vae.encode(g_low)
        except Exception:
            global_ref_latent = vae.encode(g_low[:, :, :, :3])

    # --- tile reference ---
    t_ref = _relaxed_reference_pixels(tile_pixels, tile_strength, keep_size=(qwen21 or False))
    t_latent = vae.encode(t_ref)
    ref_latents = [t_latent] if qwen21 else [global_ref_latent, t_latent]

    # --- vision slots, SAME order as reference_latents ---
    def _vision_rgb(pixels):
        samples = pixels[:1].movedim(-1, 1)
        total = int(384 * 384) if grounding_px <= 0 or grounding_px >= 384 else int(grounding_px * grounding_px)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = max(1, round(samples.shape[3] * scale_by))
        height = max(1, round(samples.shape[2] * scale_by))
        vl = comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)
        rgb = vl[:, :, :, :3]
        if vl.shape[-1] > 3:                     
            rgb = rgb * vl[:, :, :, 3:] + (1.0 - vl[:, :, :, 3:])
        return rgb

    g_rgb = _vision_rgb(global_pixels)
    t_rgb = _vision_rgb(tile_pixels)

    full_prompt = ("Picture 1: " + prompt + "\n"
                   "Picture 2: the local region of Picture 1 shown here, refine its detail "
                   "while keeping it identical to Picture 1.")
    positive = clip.encode_from_tokens_scheduled(
        clip.tokenize(full_prompt, images=[g_rgb, t_rgb], prevent_empty_text=True))

    extra = {"reference_latents": ref_latents}
    if reference_method is not None:
        extra["reference_latents_method"] = reference_method
    positive = node_helpers.conditioning_set_values(positive, extra, append=True)

    negative = None
    if encode_negative:
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize("", images=[g_rgb, t_rgb], prevent_empty_text=True))
        negative = node_helpers.conditioning_set_values(negative, extra, append=True)

    def _to_cpu_deep(v):
        if torch.is_tensor(v):
            return v.to("cpu")
        if isinstance(v, list):
            return [_to_cpu_deep(t) for t in v]
        return v

    park = lambda cond: [[e[0].to("cpu"), {k: _to_cpu_deep(v) for k, v in e[1].items()}] for e in cond]
    return park(positive), (park(negative) if negative is not None else None)


# ---------------------------------------------------------------------------
# Generic tile plumbing
# ---------------------------------------------------------------------------

def _plan_axis(size, tile, overlap):
    """Lay out tiles along one axis, evenly spaced so every gap between neighbours is identical."""
    tile = min(tile, size)
    if size <= tile:
        return [(0, size)]
    stride = max(1, tile - max(0, overlap))
    count = math.ceil((size - tile) / stride) + 1
    span = size - tile
    return [(round(i * span / (count - 1)), tile) for i in range(count)]


def _axis_fades(positions):
    """Per-tile (fade_start, fade_end) widths: the actual overlap with each neighbour."""
    fades = []
    for i, (pos, length) in enumerate(positions):
        start = (positions[i - 1][0] + positions[i - 1][1]) - pos if i > 0 else 0
        end = (pos + length) - positions[i + 1][0] if i < len(positions) - 1 else 0
        fades.append((max(0, start), max(0, end)))
    return fades


def _axis_ramp(size, fade_start, fade_end, device, dtype):
    """1-D weight ramp: 1.0 in the tile's core, fading to 0 across each overlapping edge."""
    ramp = torch.ones(size, device=device, dtype=dtype)
    if fade_start + fade_end > size:
        scale = size / (fade_start + fade_end)
        fade_start = int(fade_start * scale)
        fade_end = int(fade_end * scale)
    if fade_start > 0:
        ramp[:fade_start] *= torch.linspace(0.0, 1.0, fade_start, device=device, dtype=dtype)
    if fade_end > 0:
        ramp[-fade_end:] *= torch.linspace(1.0, 0.0, fade_end, device=device, dtype=dtype)
    return ramp


def _split_frequencies(image, radius):
    """Split a BHWC image into (low, high) about `radius` pixels, so high = image - low."""
    radius = max(1, int(radius))
    bchw = image.permute(0, 3, 1, 2)
    h, w = bchw.shape[-2:]
    small = F.interpolate(bchw, size=(max(1, h // radius), max(1, w // radius)), mode="area")
    low = F.interpolate(small, size=(h, w), mode="bicubic", align_corners=False)
    low = low.permute(0, 2, 3, 1)
    return low, image - low


def _detail_weight(weight, weight_max):
    """Near-exclusive blend weight for the detail band, so doubled contours cannot survive."""
    relative = weight / weight_max.clamp(min=1e-8)
    return ((relative - 1.0) * _DETAIL_SHARPNESS + 1.0).clamp(0.0, 1.0)


def _tile_weight(info, feather, device, dtype):
    """2-D weight map for one tile, as the outer product of its two axis ramps."""
    ramp_y = _axis_ramp(info["height"], round(info["fade_top"] * feather), round(info["fade_bottom"] * feather), device, dtype)
    ramp_x = _axis_ramp(info["width"], round(info["fade_left"] * feather), round(info["fade_right"] * feather), device, dtype)
    return (ramp_y.unsqueeze(1) * ramp_x.unsqueeze(0)).unsqueeze(0).unsqueeze(-1)


def _resolve_geometry(width, height, tile_size, overlap, alignment):
    """Fill in tile_size/overlap when left at 0, snapped to the token grid."""
    auto_tile, auto_overlap = tile_size <= 0, overlap <= 0
    if auto_tile:
        tile_size = min(_AUTO_TILE_MAX, max(_AUTO_TILE_MIN, max(width, height) // 4))
        tile_size = max(alignment, round(tile_size / alignment) * alignment)
    if auto_overlap:
        overlap = round(tile_size * 2 * _EDGE_FALLOFF / alignment) * alignment
        overlap = min(overlap, 256)   #重叠区最大 256px
    overlap = max(0, min(overlap, tile_size - alignment))
    return tile_size, overlap, auto_tile, auto_overlap


def _plan_tiles(img_h, img_w, tile_size, overlap):
    """Full tile grid over an image, with per-side overlap widths already resolved."""
    rows = _plan_axis(img_h, tile_size, overlap)
    cols = _plan_axis(img_w, tile_size, overlap)
    row_fades = _axis_fades(rows)
    col_fades = _axis_fades(cols)
    tiles = []
    tile_id = 0
    for r, (y, h) in enumerate(rows):
        for c, (x, w) in enumerate(cols):
            tiles.append({
                "tile_id": tile_id, "row": r, "col": c, "x": x, "y": y, "width": w, "height": h,
                "fade_top": row_fades[r][0], "fade_bottom": row_fades[r][1],
                "fade_left": col_fades[c][0], "fade_right": col_fades[c][1],
            })
            tile_id += 1
    return tiles, len(cols), len(rows)


def _match_channels(image, channels):
    """
    Reconcile a decoded tile with the channel count of the working image. The Qwen Image 2.1 VAE
    is RGBA and decodes four channels even for RGB sources (alpha comes back as opaque filler),
    while the canvas has as many channels as the source image had. Trim a surplus alpha or pad
    one on, otherwise the blend would see mismatched shapes.
    """
    if image.shape[-1] == channels:
        return image
    if image.shape[-1] > channels:
        return image[:, :, :, :channels]
    pad = torch.ones(image.shape[0], image.shape[1], image.shape[2], channels - image.shape[-1],
                     device=image.device, dtype=image.dtype)
    return torch.cat([image, pad], dim=-1)


def _relaxed_reference_pixels(pixels, strength, keep_size=False):
    """构造"放松"的参考。两种形态对应两类消费者：

    keep_size=False（默认，reference-latent 消费）：低通后真正缩小到 strength
        比例再返回小图（area 平均池化，天然抗混叠，无锯齿）——维持 2509/2511
        "喂小图、模型自由重新放大"的原设计，token 数随 strength 减少带来的
        自由度也一并保留。
    keep_size=True（vision/grounding、qwen21 tile 参考消费）：纯高斯低通、
        不改变尺寸（σ = 0.8(1-s)/s，s=0.5 → σ≈0.8px，与半分辨率截止感对齐；
        s<0.5 比旧重采样更平滑——旧值在那里叠了混叠锯齿，不值得对齐）。
        无重采样，锯齿从机制上不存在。
    """
    if strength >= 0.999:
        return pixels

    if not keep_size:
        bchw = pixels.permute(0, 3, 1, 2).float()
        _, _, h, w = bchw.shape
        nh, nw = max(16, round(h * strength)), max(16, round(w * strength))
        return F.interpolate(bchw, size=(nh, nw), mode="area"
                             ).to(pixels.dtype).permute(0, 2, 3, 1)

    sigma = (1.0 - strength) / max(float(strength), 0.05) * 0.8
    if sigma < 0.3:
        return pixels
    radius = max(1, int(sigma * 3))
    bchw = pixels.permute(0, 3, 1, 2).float()
    b, c, h, w = bchw.shape
    radius = min(radius, min(h, w) // 2 - 1)          # reflect pad 上限保护
    if radius < 1:
        return pixels
    x = torch.arange(-radius, radius + 1, device=pixels.device, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / k.sum()
    k = k[: 2 * radius + 1]
    k = k / k.sum()
    flat = bchw.reshape(b * c, 1, h, w)
    flat = F.pad(flat, (radius, radius, 0, 0), mode="reflect")
    flat = F.conv2d(flat, k.view(1, 1, -1, 1))
    flat = F.pad(flat, (0, 0, radius, radius), mode="reflect")
    flat = F.conv2d(flat, k.view(1, 1, 1, -1))
    out = flat.reshape(b, c, h, w).clamp(0.0, 1.0).to(pixels.dtype)
    return out.permute(0, 2, 3, 1)

def _sanitize_latent(samples):
    """NaN/Inf 单元在任何内容下都非法，会被 VAE 解码成实心色块。
    仅替换非有限值，其余数据零触碰——对所有模型、所有图像构造上安全。"""
    if torch.is_floating_point(samples) and not torch.isfinite(samples).all():
        return torch.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return samples


