# nodes_encode.py
"""ElementTileEncode — 把 ElementTileSplit 输出的 tile 图像/提示词编码为
per-tile LATENT + CONDITIONING，供原生 KSampler 逐 tile 去噪。

三条条件分支与原 TiledUpscale.refine 一致：
- Qwen Edit 类 tokenizer（图像槽）→ grounded conditioning
- Krea 2 identity-edit → in-context reference_latents
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
    _is_krea2,
    _has_krea2_in_context_patch,
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
            },
            "optional": {
                "reference_mode": ("BOOLEAN", {"default": False,
                                               "label_on": "Krea 2 identity-edit",
                                               "label_off": "conditioning"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32,
                                         "tooltip": "Vision-slot size for the per-tile text encode. "
                                                    "Qwen Edit 2509/2511: area bucket, 0 or >=384 = official 384x384 resize. "
                                                    "Qwen Image 2.1: ignored. Krea 2: caps the vision input's longest side."}),
            },
        }

    def encode(self, model, vae, clip, tile_images, tile_prompts, tile_negatives,
               reference_mode=False, grounding_px=768):
        model = _unwrap(model)
        vae = _unwrap(vae)
        clip = _unwrap(clip)
        reference_mode = bool(_unwrap(reference_mode, False))
        grounding_px = int(_unwrap(grounding_px, 768) or 0)

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

        alignment = _pixel_alignment(model)

        krea2 = _is_krea2(model.get_model_object("diffusion_model"))
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
            grounded, reference_method = False, _reference_method(model, [])
            if _is_moss_image(model):
                mode = "conditioning (MOSS-Image)"
                logging.info("[TileEncode] MOSS-Image detected; references ride as native "
                             "reference latents (index_timestep_zero_text_start). "
                             "Per-step reference_strength works on both ref_kv and ref_latents.")
            elif any(p.strip() for p in prompts) and _tokenizer_handles_images(clip) \
                    and not _is_flux2_family(model):
                grounded = True
                mode = "conditioning (grounded)"

        qwen21 = False
        if grounded:
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
            tile_latents.append({"samples": latent})

            p, ng = prompts[idx], negatives[idx]
            if grounded:
                pos, neg = _grounded_reference_conditioning(
                    clip, p, vae, crop, grounding_px,
                    reference_method=reference_method, encode_negative=True,
                    qwen21=qwen21, negative_prompt=ng, krea2=krea2)
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
