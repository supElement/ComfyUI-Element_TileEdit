# nodes_split.py
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
    _is_krea2,
    _grounded_reference_conditioning,
    _grounded_conditioning,
    _resolve_geometry,
    _plan_tiles,
    _plan_tiles_count,
)


GRID_TYPE = "ELEMENT_TILE_GRID"


def _upscale(image, factor, alignment):
    if factor == 1.0:
        return image
    _, h, w, _ = image.shape
    new_h = max(alignment, round(h * factor / alignment) * alignment)
    new_w = max(alignment, round(w * factor / alignment) * alignment)
    bchw = image.permute(0, 3, 1, 2)
    bchw = comfy.utils.common_upscale(bchw, new_w, new_h, "lanczos", "disabled")
    return bchw.permute(0, 2, 3, 1)


class ElementTileSplit:
    """Upscale + tile an image, emitting per-tile LATENT + per-tile CONDITIONING
    so the sampling itself can be done by ComfyUI's native KSampler."""

    CATEGORY = "Element TileUpscale"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", GRID_TYPE, "IMAGE")
    RETURN_NAMES = ("tile_positives", "tile_negatives","tile_latents", 
                    "grid", "tile_sources")
    OUTPUT_IS_LIST = (True, True, True, False, True)
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "tile_mode": ("BOOLEAN", {"default": True, "label_on": "by count", "label_off": "by size",
                                          "tooltip": "On: grid driven by columns/rows. Off: grid driven by tile_size."}),
                "columns": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1, "tooltip": "Tile columns across the width."}),
                "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1, "tooltip": "Tile rows down the height."}),

                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                    "tooltip": "Resolution each tile is regenerated at. 0 picks one from the output size: "
                               "about four tiles along the longest side, kept between 1024 and 1536."
                               "keep each tile ≤ 2MP for identity-edit LoRAs."}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16,
                    "tooltip": "How much neighbouring tiles share. 0 = auto: 15% of the tile size, capped "
                               "at 256px (whichever is smaller). Any non-zero value is used as-is, with no "
                               "cap. Higher = smoother seams but more compute."}),
                "reference_mode": ("BOOLEAN", {"default": False,
                    "label_on": "Krea 2 identity-edit", "label_off": "conditioning"}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Vision-slot size for the per-tile text encode. Qwen Edit 2509/2511: area "
                               "bucket, 0 or >=384 = official 384x384 resize. Qwen Image 2.1: ignored. "
                               "Krea 2: caps the vision input's longest side, 0 = tile size."}),
            },
            "optional": {
                "prompt": ("STRING", {"default": "enhance quality preserving all the details",
                                       "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True,
                                               "tooltip": "Leave blank = Encode an empty string as the baseline negative condition."}),
            },
        }

    
    def split(self, model, image, vae, upscale_by, tile_size, overlap, reference_mode, grounding_px, clip=None, 
              prompt="enhance quality preserving all the details", negative_prompt="", tile_mode=True, columns=2, rows=2):
        if clip is None:
            raise ValueError(
                "ElementTileSplit:'clip' is not connected. The positive/negative input ports have been removed."
                "Per-tile conditioning is encoded using CLIP + prompt/negative_prompt.")
        if negative_prompt is None:
            negative_prompt = ""
        alignment = _pixel_alignment(model)
        if reference_mode and not _is_krea2(model.get_model_object("diffusion_model")):
            logging.warning("[TileSplit] reference_mode is Krea 2 identity-edit only; "
                            "falling back to the conditioning path (pair with "
                            "ElementStrengthSchedule for per-step reference control).")
            reference_mode = False
        if reference_mode:
            mode = "in_context"
            grounded = False
            source_holder = None
            reference_method = None
        else:
            mode = "conditioning"
            grounded = False
            reference_method = _reference_method(model, [])
            if clip is not None and prompt and _tokenizer_handles_images(clip) \
                    and not _is_flux2_family(model):
                grounded = True
                mode = "conditioning (grounded)"
    
        source = _upscale(image, upscale_by, alignment)
        _, img_h, img_w, _ = source.shape
        pad_h, pad_w = (-img_h) % alignment, (-img_w) % alignment
        if pad_h or pad_w:
            bchw = source.permute(0, 3, 1, 2)
            bchw = torch.nn.functional.pad(bchw, (0, pad_w, 0, pad_h), mode="replicate")
            source = bchw.permute(0, 2, 3, 1)
        work_h, work_w = img_h + pad_h, img_w + pad_w
        if tile_mode:
            tiles, num_x, num_y, tile_size, overlap = _plan_tiles_count(
                work_h, work_w, columns, rows, alignment, overlap)
        else:
            tile_size, overlap, _, _ = _resolve_geometry(
                work_w, work_h, tile_size, overlap, alignment)
            tiles, num_x, num_y = _plan_tiles(work_h, work_w, tile_size, overlap)

        total_tiles = len(tiles)
        pbar = comfy.utils.ProgressBar(total_tiles)
        logging.info("[TileSplit] %s | %d tile(s) | tile=%dx%d overlap=%d",
                     mode, total_tiles, tile_size, tile_size, overlap)
    
        if grounded:
            base_pos = base_neg = None
        else:
            base_pos = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
            base_neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
    
        tile_pos, tile_neg, tile_latents, tile_src = [], [], [], []
        qwen21 = False
        if grounded:
            tok_name = type(getattr(clip, "tokenizer", None)).__name__.lower().replace("_", "")
            qwen21 = ("qwenimage21" in tok_name) and alignment == 32
    
        for index, info in enumerate(tiles):
            pbar.update_absolute(index, total_tiles)
            x, y, w, h = info["x"], info["y"], info["width"], info["height"]
            logging.info("[TileSplit] encoding tile %d/%d (row=%d col=%d) at (%d,%d) size %dx%d",
                         index + 1, total_tiles, info["row"], info["col"], info["x"], info["y"], w, h)
            crop = source[:, y:y+h, x:x+w, :].clone()
            tile_src.append(crop)
    
            # -- source latent --
            try:
                tile_latent = vae.encode(crop)
            except Exception:
                tile_latent = vae.encode(crop[:, :, :, :3])
            tile_latent = tile_latent.to(
                device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype())
            tile_latent = comfy.sample.fix_empty_latent_channels(model, tile_latent)
            tile_latents.append({"samples": tile_latent})
    
            # -- conditioning --
            if grounded:
                pos, neg = _grounded_reference_conditioning(
                    clip, prompt, vae, crop, grounding_px,
                    reference_method=reference_method,
                    encode_negative=True,
                    qwen21=qwen21,
                    negative_prompt=negative_prompt)            
            elif reference_mode:
                ref_latent = vae.encode(crop)                   
                extra = {"reference_latents": [ref_latent], "reference_latents_method": "index_timestep_zero"}
                if _tokenizer_handles_images(clip):
                    pos = _grounded_conditioning(clip, prompt, crop, grounding_px)                  
                    neg = _grounded_conditioning(clip, negative_prompt or "", crop, grounding_px)   
                else:
                    pos, neg = base_pos, base_neg
                pos = node_helpers.conditioning_set_values(pos, extra, append=True)
                neg = node_helpers.conditioning_set_values(neg, extra, append=True)
            else:
                # generic reference-latent path：静态放松已移除，参考 = 原始 tile latent，
                # 逐步削弱由采样模型上的调度 patch（ElementStrengthSchedule）在前向内完成
                ref_latent = tile_latent
                pos = node_helpers.conditioning_set_values(
                    base_pos, {"reference_latents": [ref_latent]}, append=True)
                if reference_method is not None:
                    pos = node_helpers.conditioning_set_values(
                        pos, {"reference_latents_method": reference_method}, append=True)
                neg = base_neg
            tile_pos.append(pos)
            tile_neg.append(neg)
            pbar.update_absolute(index + 1, total_tiles)
    
        grid = {
            "tiles": tiles, "num_x": num_x, "num_y": num_y,
            "img_h": img_h, "img_w": img_w, "work_h": work_h, "work_w": work_w,
            "overlap": overlap, "tile_size": tile_size, "alignment": alignment,
            "channels": source.shape[-1], "reference_mode": reference_mode,
        }
        return (tile_pos, tile_neg, tile_latents, grid, tile_src)
    
    
    