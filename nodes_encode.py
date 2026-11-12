# nodes_encode.py
"""ElementTileEncode — 把 ElementTileSplit 输出的 tile 图像/提示词编码为
per-tile LATENT + CONDITIONING，供原生 KSampler 逐 tile 去噪。

三条条件分支与原 TiledUpscale.refine 一致：
- Qwen Edit 类 tokenizer（图像槽）→ grounded conditioning
- Krea 2 identity-edit → in-context reference_latents
- MiniMax-H3-Image → minimax_refs 参考通道 + 1024 短边 vision（Picture 1）
- 其它模型 → 通用 reference_latents（静态参考；逐步削弱交给 ElementStrengthSchedule）
"""

import logging
import torch
import comfy.utils
import comfy.model_management
import node_helpers
from .common import (
    _pixel_alignment,
    _reference_method,
    _tokenizer_handles_images,
    _is_flux2_family,
    _is_moss_image,
    _is_minimax_image,
    _is_krea2,
    _has_krea2_in_context_patch,
    _model_family,
    _strip_owned_patches,
    _grounded_reference_conditioning,
    _grounded_conditioning,
)
from .nodes_patch import _in_context_model_from_conditioning


def _unwrap(value, default=None):
    """INPUT_IS_LIST=True 时 ComfyUI 会把每个输入包一层 list；标量取第一个。"""
    if value is None:
        return default
    if isinstance(value, list):
        if len(value) > 1:
            logging.warning("[TileEncode] multiple items on a scalar input; using the first.")
        return value[0] if len(value) else default
    return value


def _as_string_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return ["" if s is None else str(s) for s in v]


class ElementTileEncode:
    CATEGORY = "Element TileEdit"
    FUNCTION = "encode"
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("model", "tile_positives", "tile_negatives", "tile_latents")
    OUTPUT_IS_LIST = (False, True, True, True)
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "tile_images": ("IMAGE",),
                "tile_prompts": ("STRING", {"forceInput": True}),
                "tile_negatives": ("STRING", {"forceInput": True}),
                "tile_masks": ("MASK", {"tooltip": "Required: ElementTileSplit's 6th output (tile_masks). "
                                                   "White = repaint, black = keep as reference. "
                                                   "When Split's edge mask is off this is an all-white list, so wiring it is always safe; "
                                                   "leaving it unconnected while the edge mask is on silently disables the sampler lock."}),
            },
            "optional": {
                "reference_mode": ("BOOLEAN", {"default": False,
                                               "label_on": "Krea 2 identity-edit",
                                               "label_off": "conditioning"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32,
                                         "tooltip": "Vision-slot size for the per-tile text encode. "
                                                    "Qwen Edit 2509/2511: area bucket, 0 or >=384 = official 384x384 resize. "
                                                    "Qwen Image 2.1: ignored. Krea 2: caps the vision input's longest side. "
                                                    "MiniMax-H3-Image: vision short edge, 0 = official 1024 (the VAE reference stays at core's 1024-2048 short edge)."}),
            },
        }

    def encode(self, model, vae, clip, tile_images, tile_prompts, tile_negatives,
               reference_mode=False, grounding_px=768, tile_masks=None):
        model = _unwrap(model)
        vae = _unwrap(vae)
        clip = _unwrap(clip)
        reference_mode = bool(_unwrap(reference_mode, False))
        grounding_px = int(_unwrap(grounding_px, 768) or 0)
        family = _model_family(model)

        prompts = _as_string_list(tile_prompts)
        negatives = _as_string_list(tile_negatives)
        images = list(tile_images) if isinstance(tile_images, (list, tuple)) else [tile_images]

        n = len(images)
        if n == 0:
            raise ValueError("ElementTileEncode: no tile images received.")
        
        if len(prompts) != n:
            raise ValueError(f"ElementTileEncode: {len(prompts)} prompt(s) for {n} tile image(s).")
        if len(negatives) != n:
            raise ValueError(f"ElementTileEncode: {len(negatives)} negative(s) for {n} tile image(s).")
        if clip is None:
            raise ValueError("ElementTileEncode: 'clip' is not connected.")

        masks = []
        if tile_masks is not None:
            masks = list(tile_masks) if isinstance(tile_masks, (list, tuple)) else [tile_masks]
            if len(masks) != n:
                logging.warning("[TileEncode] %d tile mask(s) for %d tile image(s); masks ignored.",
                                len(masks), n)
                masks = []
        else:
            logging.warning("[TileEncode] tile_masks is not connected; sampler-side lock is disabled "
                            "even if Split's edge mask is on.")

        alignment = _pixel_alignment(model)

        auto = family == "auto"
        diffusion_model = model.get_model_object("diffusion_model")
        if auto:
            krea2 = _is_krea2(diffusion_model)
            minimax = _is_minimax_image(model)
            moss = _is_moss_image(model)
            flux_latent = _is_flux2_family(model)
        else:
            krea2 = family == "krea2"
            minimax = family == "minimax"
            moss = family == "moss"
            flux_latent = family in ("flux2", "flux3")
            logging.info("[TileEncode] model family from ElementStrengthSchedule: %s", family)
        has_in_context_patch = krea2 and _has_krea2_in_context_patch(model)
        in_context = krea2 and (reference_mode or has_in_context_patch)
        if reference_mode and not krea2:
            logging.warning("[TileEncode] reference_mode is Krea 2 identity-edit only; "
                            "falling back to the conditioning path (pair with "
                            "ElementStrengthSchedule for per-step reference control).")

        if in_context:
            mode = "in_context"
            grounded, reference_method = False, None
        else:
            mode = "conditioning"
            grounded = False
            if family == "moss":
                reference_method = "index_timestep_zero_text_start"
            elif family == "flux2":
                reference_method = "index"
            else:
                reference_method = _reference_method(model, [])
            if moss:
                mode = "conditioning (MOSS-Image)"
                logging.info("[TileEncode] MOSS-Image detected; references ride as native "
                             "reference latents (index_timestep_zero_text_start). "
                             "Per-step reference_strength works on both ref_kv and ref_latents.")
            elif minimax:
                grounded = True
                mode = "conditioning (MiniMax-H3-Image)"
                logging.info("[TileEncode] MiniMax-H3-Image detected; each tile rides as "
                             "Picture 1 (minimax_refs + vision) and is re-injected every step.")
            elif family in ("krea2", "qwen21", "qwen_edit") or (
                    auto and any(p.strip() for p in prompts)
                    and _tokenizer_handles_images(clip) and not flux_latent):
                grounded = True
                mode = "conditioning (grounded)"

        qwen21 = False
        if grounded:
            if family == "qwen21":
                qwen21 = True
            elif family == "qwen_edit":
                qwen21 = False
            else:
                tok_name = type(getattr(clip, "tokenizer", None)).__name__.lower().replace("_", "")
                qwen21 = ("qwenimage21" in tok_name) and alignment == 32

        pbar = comfy.utils.ProgressBar(n)
        logging.info("[TileEncode] %s | %d tile(s) | alignment=%d", mode, n, alignment)

        base_cache = {}

        def _base_conditioning(text):
            if text not in base_cache:
                base_cache[text] = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            return base_cache[text]

        tile_pos, tile_neg, tile_latents = [], [], []
        for idx in range(n):
            pbar.update_absolute(idx, n)
            crop = images[idx]
            if crop.ndim == 3:
                crop = crop.unsqueeze(0)
            crop = crop[:1]
            logging.info("[TileEncode] encoding tile %d/%d (%dx%d)",
                         idx + 1, n, crop.shape[2], crop.shape[1])

            try:
                latent = vae.encode(crop)
            except Exception:
                latent = vae.encode(crop[:, :, :, :3])
            latent = latent.to(device=comfy.model_management.intermediate_device(),
                               dtype=comfy.model_management.intermediate_dtype())
            latent = comfy.sample.fix_empty_latent_channels(model, latent)
            latent_out = {"samples": latent}
            if idx < len(masks) and masks[idx] is not None:
                m = masks[idx]
                if m.ndim == 4:
                    m = m[:1, :1]
                elif m.ndim == 3:
                    m = m[:1].unsqueeze(1)
                elif m.ndim == 2:
                    m = m[None, None]
                # 全白 = 整块都可重绘，无需 inpaint 路径
                if float(m.min()) < 1.0:
                    latent_out["noise_mask"] = m.to(device=latent.device, dtype=torch.float32)
            tile_latents.append(latent_out)

            p, ng = prompts[idx], negatives[idx]
            if grounded:
                pos, neg = _grounded_reference_conditioning(
                    clip, p, vae, crop, grounding_px,
                    reference_method=reference_method, encode_negative=True,
                    qwen21=qwen21, negative_prompt=ng, krea2=krea2, minimax=minimax)
            elif in_context:
                ref_latent = vae.encode(crop)
                extra = {"reference_latents": [ref_latent],
                         "reference_latents_method": "index_timestep_zero"}
                if _tokenizer_handles_images(clip):
                    pos = _grounded_conditioning(clip, p, crop, grounding_px)
                    neg = _grounded_conditioning(clip, ng or "", crop, grounding_px)
                else:
                    pos, neg = _base_conditioning(p), _base_conditioning(ng)
                pos = node_helpers.conditioning_set_values(pos, extra, append=True)
                neg = node_helpers.conditioning_set_values(neg, extra, append=True)
            else:
                # 通用 reference-latent：参考 = 本 tile latent，逐步削弱由
                # ElementStrengthSchedule 的模型 patch 在前向内完成
                pos = node_helpers.conditioning_set_values(
                    _base_conditioning(p), {"reference_latents": [latent]}, append=True)
                if reference_method is not None:
                    pos = node_helpers.conditioning_set_values(
                        pos, {"reference_latents_method": reference_method}, append=True)
                neg = _base_conditioning(ng)
            tile_pos.append(pos)
            tile_neg.append(neg)
            pbar.update_absolute(idx + 1, n)

        if in_context and not has_in_context_patch:
            model = _strip_owned_patches(model, keys=("tiled_upscale_in_context_cond",
                                                      "tiled_upscale_in_context"))
            model = _in_context_model_from_conditioning(model)

        return (model, tile_pos, tile_neg, tile_latents)
