import torch.nn.functional as F
import torch
import math
import logging
from einops import rearrange

import latent_preview

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import comfy.patcher_extension
import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
import node_helpers
import inspect

# Tile dimensions are snapped to a multiple of the model's pixels-per-token so a tile maps to a
# whole number of tokens. 16 for FLUX.2 (16x VAE, patch 1) and Krea 2 (8x VAE, patch 2), 16 for
# Qwen-Image / Edit 2509 / 2511 (8x VAE, patch 2), 32 for Qwen Image 2.1 (16x VAE, patch 2). Read
# from the loaded model by _pixel_alignment(); this is only the fallback.
LATENT_SCALE = 16

# Fraction of an overlap band over which a tile's *input* fades from the already-generated
# neighbour back to the plain source. Deliberately small: the neighbour's pixels have to reach the
# model at full strength across most of the band, or the model stops treating them as something to
# continue. The fade only exists so the input has no step at the rim.
_CONTEXT_EDGE = 0.25

# How far a tile drifts from its neighbours near its own edges, as a fraction of the tile.
_EDGE_FALLOFF = 0.15

# Bounds for an automatically chosen tile. Attention cost grows with the square of the token count
# while the tile count only falls linearly with area, so the total cost grows quadratically.
_AUTO_TILE_MIN, _AUTO_TILE_MAX = 1024, 1536

# How much narrower the detail band's transition is than the blend ramp, so doubled contours
# cannot survive the blend.
_DETAIL_SHARPNESS = 20.0

# Frequency split between the two, as a fraction of the overlap.
_DETAIL_RADIUS = 0.25


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


def _grounded_conditioning(clip, prompt, image, grounding_px):
    """
    Encode the prompt with one tile as the text encoder's vision input. The Krea 2 edit LoRAs see
    the source twice during training: as latent tokens (appearance) and inside the text encoder's
    user turn (semantics). Grounding per tile keeps both paths talking about the same pixels.
    """
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

def _grounded_reference_conditioning(clip, prompt, vae, tile_pixels, grounding_px, reference_method=None, encode_negative=False, qwen21=False):
    samples = tile_pixels[:1].movedim(-1, 1)
    if qwen21:
        # Qwen Image 2.1 (official TextEncodeQwenImage21): the text encoder and the VAE see the
        # SAME image, because the DiT splices the reference latent exactly where the vision
        # tokens sat ("image_slots"), one vision token == one 32px DiT token. The tile is
        # already 32px-aligned and the sampling latent IS this size, so no resize — any other
        # size shifts the edit.
        s = tile_pixels[:1]
        rgb = s[:, :, :, :3]
        if s.shape[-1] > 3:
            rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])  # vision tower sees alpha over white, the vae keeps all four
        ref_latent = vae.encode(s)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=[rgb], prevent_empty_text=True))
    else:
        # 2509/2511 (official TextEncodeQwenImageEditPlus): 384x384-area vision bucket.
        ref_latent = vae.encode(tile_pixels[:1])
        total = int(384 * 384) if grounding_px <= 0 or grounding_px >= 384 else int(grounding_px * grounding_px)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = max(1, round(samples.shape[3] * scale_by))
        height = max(1, round(samples.shape[2] * scale_by))
        vl = comfy.utils.common_upscale(samples, width, height, "area", "disabled").movedim(1, -1)
        rgb = vl[:, :, :, :3]
        if vl.shape[-1] > 3:
            rgb = rgb * vl[:, :, :, 3:] + (1.0 - vl[:, :, :, 3:])
        positive = clip.encode_from_tokens_scheduled(clip.tokenize("Picture 1: " + prompt, images=[rgb], prevent_empty_text=True))


    extra = {"reference_latents": [ref_latent]}
    if reference_method is not None:
        extra["reference_latents_method"] = reference_method
    positive = node_helpers.conditioning_set_values(positive, extra, append=True)
    negative = None
    if encode_negative:
        negative = clip.encode_from_tokens_scheduled(clip.tokenize("", images=[rgb], prevent_empty_text=True))
        negative = node_helpers.conditioning_set_values(negative, extra, append=True)
    def _to_cpu_deep(v):
        if torch.is_tensor(v):
            return v.to("cpu")
        if isinstance(v, list):
            return [_to_cpu_deep(t) for t in v]   # int/str 等非 tensor 原样保留
        return v
    
    def park(cond):
        return [[e[0].to("cpu"), {k: _to_cpu_deep(v) for k, v in e[1].items()}] for e in cond]

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


def _tile_sampling_callback(previewer, steps, state):
    """
    Per-tile sampling callback, modelled directly on ComfyUI's own latent_preview.prepare_callback
    (the exact pattern every core sampler node uses): one progress bar per tile covering that
    tile's steps, advanced every step together with a fresh latent preview. This gives the node
    overlay the familiar 0-100% sweep during each tile's sampling.
    """
    pbar = comfy.utils.ProgressBar(steps)

    def callback(step, x0, x, total_steps):
        preview_bytes = None
        if previewer is not None:
            preview_bytes = previewer.decode_latent_to_preview_image("JPEG", x0)
        comfy.model_management.throw_exception_if_processing_interrupted()
        state["preview"] = preview_bytes
        pbar.update_absolute(step + 1, total_steps, preview_bytes)

    return callback


class TiledUpscale:
    CATEGORY = "Element TileUpscale"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "split_images", "resampler_images")
    OUTPUT_IS_LIST = (False, True, True)
    FUNCTION = "refine"
    DESCRIPTION = ("All-in-one tiled refine/upscale for reference-latent edit models (Qwen Image "
                   "2.1 / Edit 2509 / 2511, FLUX.2 klein, Krea 2 and similar). Upscales, splits "
                   "into overlapping tiles, regenerates each tile at full tile resolution and "
                   "blends everything into one image. Works with any model/positive/negative/vae "
                   "combination that is valid on its own: models without reference-latent support "
                   "(SDXL, Qwen-Image-2512, ...) simply ignore the per-tile reference and run as "
                   "tiled img2img, where feature preservation is governed by denoise alone.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16, "tooltip": "Resolution each tile is regenerated at. 0 picks one from the output size: "
                                     "about four tiles along the longest side, kept between 1024 and 1536."}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16, "tooltip": "How much neighbouring tiles share. 0 picks 30% of the tile size."}),
                "feather": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "reference_mode": ("BOOLEAN", {"default": False, "label_on": "Krea 2 identity-edit", "label_off": "conditioning", "tooltip": "Off: tile is attached as reference latent; with 'clip'+'prompt' it also enters the text encoder's vision slot. On: Krea 2 identity-edit in-context forward, requires 'clip'."}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Vision-slot size for the per-tile text encode. Qwen Edit 2509/2511: area bucket, 0 or >=384 = official 384x384 resize. Qwen Image 2.1: ignored (locked to the tile's size). Krea 2: caps the vision input's longest side, 0 = tile size."}),
                "clip": ("CLIP", {"tooltip": "Connect for vision-edit models (Qwen Image 2.1, FLUX.2 klein, Krea 2 style-ref LoRA): each tile is re-encoded with itself as the text encoder's vision input, replicating the official reference pattern per tile. When connected, the 'positive'/'negative' inputs are bypassed and 'prompt' is used instead."}),
                "prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Instruction encoded per tile together with the tile itself as vision input. Used when 'clip' is connected (in either reference_mode); leave empty to use the 'positive' input unmodified (bare reference latent, no vision grounding)."}),
            },
        }

    def _upscale(self, image, factor, alignment=LATENT_SCALE):
        if factor == 1.0:
            return image
        _, h, w, _ = image.shape
        new_h = max(alignment, round(h * factor / alignment) * alignment)
        new_w = max(alignment, round(w * factor / alignment) * alignment)
        bchw = image.permute(0, 3, 1, 2)
        bchw = comfy.utils.common_upscale(bchw, new_w, new_h, "lanczos", "disabled")
        return bchw.permute(0, 2, 3, 1)

    def _sample_tile(self, model, sampling_model, positive, negative, vae, tile_pixels, seed, steps, cfg, sampler_name, scheduler, denoise,
                     reference_method=None, source_holder=None, previewer=None, state=None, disable_pbar=True, attach_reference=True):
        """Regenerate one tile: encode it, hand it to the model as reference, sample, decode."""
        if tile_pixels.shape[-1] > 3:
            try:
                tile_latent = vae.encode(tile_pixels)
            except Exception:
                tile_latent = vae.encode(tile_pixels[:, :, :, :3])
        else:
            tile_latent = vae.encode(tile_pixels)

        if source_holder is not None:
            source_holder["latent"] = model.model.process_latent_in(tile_latent)
            positive_tile = positive
            negative_tile = negative
        elif not attach_reference:
            positive_tile = positive
            negative_tile = negative
        else:
            positive_tile = node_helpers.conditioning_set_values(positive, {"reference_latents": [tile_latent]}, append=True)
            negative_tile = negative  
            if reference_method is not None:
                positive_tile = node_helpers.conditioning_set_values(positive_tile, {"reference_latents_method": reference_method})


        latent = tile_latent.to(device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        noise = comfy.sample.prepare_noise(latent, seed, None)

        samples = comfy.sample.sample(sampling_model, noise, steps, cfg, sampler_name, scheduler, positive_tile, negative_tile,
                                      latent, denoise=denoise, disable_pbar=disable_pbar, seed=seed,
                                      callback=_tile_sampling_callback(previewer, steps, state))

        decoded = vae.decode(samples)
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        return decoded

    def refine(self, model, positive, negative, vae, image, upscale_by, tile_size, overlap, feather, seed, steps, cfg,
               sampler_name, scheduler, denoise, reference_mode=False, grounding_px=768, clip=None, prompt=""):
        alignment = _pixel_alignment(model)

        if reference_mode:
            mode = "in_context"
            grounded_conditioning = False
            if clip is None:
                raise ValueError("TiledUpscale: reference_mode (in-context, Krea 2 identity-edit) requires the 'clip' input.")
            if not prompt:
                logging.warning("[TiledUpscale] reference_mode is on but 'prompt' is empty; the model will see no instruction.")
            source_holder = {"latent": None}
            sampling_model = _in_context_model(model, source_holder)
            reference_method = None
        else:
            mode = "conditioning"
            grounded_conditioning = False
            source_holder = None
            sampling_model = model
            reference_method = _reference_method(model, positive)

            if clip is not None and prompt and _tokenizer_handles_images(clip) and not _is_flux2_family(model):
                grounded_conditioning = True
                mode = "conditioning (grounded)"

            elif clip is not None and prompt:
                logging.warning("[TiledUpscale] this model's text encoder is text-only (e.g. FLUX.2 "
                                "klein's Qwen3); the tile cannot enter a vision slot, falling back to "
                                "the reference-latent-only path with the original conditioning.")
            elif clip is not None:
                logging.warning("[TiledUpscale] 'clip' is connected but 'prompt' is empty; falling "
                                "back to reference-latent-only conditioning. For Qwen Image 2.1 / "
                                "FLUX.2 klein / Krea 2 style-ref, fill 'prompt' for correct results.")

        source = self._upscale(image, upscale_by, alignment)
        _, img_h, img_w, _ = source.shape

        pad_h = (-img_h) % alignment
        pad_w = (-img_w) % alignment
        if pad_h or pad_w:
            bchw = source.permute(0, 3, 1, 2)
            bchw = F.pad(bchw, (0, pad_w, 0, pad_h), mode="replicate")
            source = bchw.permute(0, 2, 3, 1)
        work_h, work_w = img_h + pad_h, img_w + pad_w

        tile_size, overlap, auto_tile, auto_overlap = _resolve_geometry(work_w, work_h, tile_size, overlap, alignment)
        tiles, num_x, num_y = _plan_tiles(work_h, work_w, tile_size, overlap)
        logging.info("[TiledUpscale] branch=%s | tokenizer=%s | vision=%s | %d tile(s).",
                     mode,
                     type(getattr(clip, "tokenizer", None)).__name__ if clip is not None else "none",
                     _tokenizer_handles_images(clip) if clip is not None else False,
                     len(tiles))

        device, dtype = source.device, source.dtype
        channels = source.shape[-1]

        canvas = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
        weight_sum = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)
        detail = torch.zeros_like(canvas)
        detail_sum = torch.zeros_like(weight_sum)
        weight_max = torch.zeros_like(weight_sum)
        context = torch.zeros_like(canvas)

        for info in tiles:
            region = (slice(None), slice(info["y"], info["y"] + info["height"]), slice(info["x"], info["x"] + info["width"]), slice(None))
            weight_max[region] = torch.maximum(weight_max[region], _tile_weight(info, feather, device, dtype))

        detail_radius = max(1, round(overlap * _DETAIL_RADIUS))

        def merged_image():
            return (canvas / weight_sum.clamp(min=1e-8)) + (detail / detail_sum.clamp(min=1e-8))

        redo_first = len(tiles) > 1

        previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
        state = {"preview": None}
        total_runs = len(tiles) + (1 if redo_first else 0)
        total_pbar = comfy.utils.ProgressBar(total_runs)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        grounded_in_context = mode == "in_context"
        tile_positive = tile_negative = None
        qwen21 = False
        if grounded_conditioning:
            tok_name = type(getattr(clip, "tokenizer", None)).__name__.lower().replace("_", "")
            qwen21 = ("qwenimage21" in tok_name) and alignment == 32
            logging.info("[TiledUpscale] grounding pattern: %s", "qwen-image-2.1 (native size, image_slots)" if qwen21 else "edit-plus (384 vision bucket)")
        attach_reference = True

        if grounded_in_context or grounded_conditioning:
            tile_positive, tile_negative = [], []
            encode_total = len(tiles)
            for encode_index, info in enumerate(tiles):
                logging.info("[TiledUpscale] encoding tile %d/%d (row=%d col=%d) at (%d,%d)",
                             encode_index + 1, encode_total, info["row"], info["col"],
                             info["x"], info["y"])
                crop = source[:, info["y"]:info["y"] + info["height"], info["x"]:info["x"] + info["width"], :]

                if grounded_in_context:
                    tile_positive.append(_grounded_conditioning(clip, prompt, crop, grounding_px))
                    if cfg > 1.0:
                        tile_negative.append(_grounded_conditioning(clip, "", crop, grounding_px))
                else:
                    pos_c, neg_c = _grounded_reference_conditioning(clip, prompt, vae, crop, grounding_px,
                                                                    reference_method, encode_negative=(cfg > 1.0), qwen21=qwen21)

                    tile_positive.append(pos_c)
                    if cfg > 1.0:
                        tile_negative.append(neg_c)

            if grounded_conditioning:
                attach_reference = False


        split_images = []
        for info in tiles:
            split_images.append(source[:, info["y"]:info["y"] + info["height"], info["x"]:info["x"] + info["width"], :].clone())

        tiles_out = [None] * len(tiles)

        def place_tile(info, previous=None):
            """Regenerate one tile and blend it into the canvas. Returns the generated tile."""
            x, y, w, h = info["x"], info["y"], info["width"], info["height"]
            region = (slice(None), slice(y, y + h), slice(x, x + w), slice(None))
            weight = _tile_weight(info, feather, device, dtype)
            detail_weight = _detail_weight(weight, weight_max[region])

            if previous is not None:
                low, high = _split_frequencies(previous, detail_radius)
                canvas[region] -= low * weight
                weight_sum[region] -= weight
                detail[region] -= high * detail_weight
                detail_sum[region] -= detail_weight
                context[region] -= previous * weight

            source_crop = source[region]

            done = weight_sum[region]
            blended = context[region] / done.clamp(min=1e-8)
            alpha = (done / _CONTEXT_EDGE).clamp(0.0, 1.0)
            tile_input = blended * alpha + source_crop * (1.0 - alpha)

            generated = self._sample_tile(model, sampling_model,
                                          tile_positive[info["tile_id"]] if tile_positive else positive,
                                          tile_negative[info["tile_id"]] if tile_negative else negative,
                                          vae, tile_input, seed, steps, cfg, sampler_name, scheduler, denoise,
                                          reference_method, source_holder, previewer, state, disable_pbar,
                                          attach_reference)

            if generated.shape[1] != h or generated.shape[2] != w:
                generated = F.interpolate(generated.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            generated = generated.to(device=device, dtype=dtype)
            generated = _match_channels(generated, channels)

            low, high = _split_frequencies(generated, detail_radius)
            canvas[region] += low * weight
            weight_sum[region] += weight
            detail[region] += high * detail_weight
            detail_sum[region] += detail_weight
            context[region] += generated * weight
            tiles_out[info["tile_id"]] = generated
            return generated

        first_tile = None
        for index, info in enumerate(tiles):
            total_pbar.update_absolute(index, total_runs, state["preview"])
            logging.info("[TiledUpscale] tile %d/%d (row=%d col=%d) at (%d,%d) size %dx%d",
                         index + 1, total_runs, info["row"], info["col"], info["x"], info["y"],
                         info["width"], info["height"])
            generated = place_tile(info)
            total_pbar.update_absolute(index + 1, total_runs, state["preview"])
            if index == 0 and redo_first:
                first_tile = generated


        if redo_first:
            total_pbar.update_absolute(len(tiles), total_runs, state["preview"])
            info = tiles[0]
            logging.info("[TiledUpscale] tile %d/%d (row=%d col=%d) at (%d,%d) size %dx%d (redo with finished neighbours)",
                         total_runs, total_runs, info["row"], info["col"], info["x"], info["y"],
                         info["width"], info["height"])
            place_tile(tiles[0], previous=first_tile)
            total_pbar.update_absolute(total_runs, total_runs, state["preview"])


        merged = merged_image()[:, :img_h, :img_w, :]
        return (merged.clamp(0.0, 1.0), split_images, tiles_out)
