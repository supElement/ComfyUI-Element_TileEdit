# common.py
import math
import logging
import inspect
import torch
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
    if "flux" in dm_name:
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
    """
    Krea 2's forward with the source latent prepended as a block of clean tokens. ComfyUI's
    native Krea 2 forward is text-to-image: `[text | target]`, with reference latents appended
    *after* the target and given their own timestep-zero modulation. The Krea 2 edit LoRAs are
    trained the other way round, on `[text | source(frame=1) | target(frame=0)]` with one shared
    timestep, the source told apart from the noisy target only by its RoPE frame index, so the
    native path produces noise with them. This rebuilds the sequence the way they expect.
    """
    patch = dm.patch
    temporal = x.ndim == 5
    if temporal:
        b5, _, t5, _, _ = x.shape
        x = _to_4d(x)
    batch, _, h_orig, w_orig = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    height, width = x.shape[-2:]
    h_, w_ = height // patch, width // patch

    src = _to_4d(source).to(device=x.device, dtype=x.dtype)
    if src.shape[0] != batch:
        src = src[:1].expand(batch, *src.shape[1:])
    if src.shape[-2:] != (height, width):
        src = F.interpolate(src.float(), size=(height, width), mode="bilinear").to(x.dtype)
    src = comfy.ldm.common_dit.pad_to_patch_size(src, (patch, patch))

    to_tokens = lambda t: dm.first(rearrange(t, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))
    target_tokens = to_tokens(x)
    source_tokens = to_tokens(src)

    t = dm.tmlp(timestep_embedding(timesteps, dm.tdim).unsqueeze(1).to(target_tokens.dtype))
    tvec = dm.tproj(t)

    context = dm.txtfusion(dm._unpack_context(context), mask=None, transformer_options=transformer_options)
    context = dm.txtmlp(context)

    txt_len, src_len, tgt_len = context.shape[1], source_tokens.shape[1], target_tokens.shape[1]
    combined = torch.cat([context, source_tokens, target_tokens], dim=1)

    device = combined.device
    pos = torch.cat([torch.zeros(batch, txt_len, 3, device=device, dtype=torch.float32),
                     _img_ids(batch, 1, h_, w_, device),
                     _img_ids(batch, 0, h_, w_, device)], dim=1)
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


def _in_context_model(model, source_holder):
    """
    Clone the model with a forward that reads the current tile out of `source_holder`. Cloned once
    rather than per tile: a fresh ModelPatcher would make ComfyUI re-apply every LoRA patch for
    each tile, and these models are always LoRA-patched.
    """
    patched = model.clone()

    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        source = source_holder["latent"]
        if source is None:
            return executor(x, timesteps, context, *args, **kwargs)
        options = kwargs.get("transformer_options")
        if options is None:
            options = next((arg for arg in reversed(args) if isinstance(arg, dict)), {})
        return _krea2_in_context_forward(executor.class_obj, x, timesteps, context, source, options)

    options = patched.model_options.setdefault("transformer_options", {})
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "tiled_upscale_in_context", wrapper, options)
    return patched


_GROUNDING_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)


def _grounded_conditioning(clip, prompt, image, grounding_px, reference_strength=1.0):
    """
    Encode the prompt with one tile as the text encoder's vision input. The Krea 2 edit LoRAs see
    the source twice during training: as latent tokens (appearance) and inside the text encoder's
    user turn (semantics). Grounding per tile keeps both paths talking about the same pixels.
    """
    image = _relaxed_reference_pixels(image, reference_strength, keep_size=True)
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
                                     reference_method=None, encode_negative=False,
                                     qwen21=False, reference_strength=1.0,
                                     global_pixels=None, global_ref_latent=None,
                                     negative_prompt=""):
    samples = tile_pixels[:1].movedim(-1, 1)
    dual = global_pixels is not None and global_ref_latent is not None
    if qwen21:
        # Qwen Image 2.1: 文本编码器把 vision token 从 conditioning 中丢弃, 只记录位置 image_slots;
        # DiT 在这些位置拼接 reference latent (build_sequence)。参考 latent 的尺寸是自由的——每个
        # ref 贡献的 token 数由它自己的 h,w 决定——但官方节点保持"第一张参考 = 输出尺寸"(其它尺寸
        # 会移动编辑结果), 且每张参考的 vision 图与 VAE latent 必须来自同一份像素(槽位跨度与拼接块
        # 对齐)。因此: tile 参考放第一、保持 tile 原尺寸; 全局参考放第二、用它自己更小的尺寸。
        s = _relaxed_reference_pixels(tile_pixels[:1], reference_strength, keep_size=True)
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
        # 2509/2511 的参考 latent 是多图参考之一, 尺寸可与输出不同,
        # 直接喂一张真正缩小过的参考, 模型会自由地重新放大它。
        ref_latents = [vae.encode(_relaxed_reference_pixels(tile_pixels[:1], reference_strength))]
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
        if vl.shape[-1] > 3:                      # 检查切之前的通道数
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



