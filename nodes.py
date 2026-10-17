import logging

import torch
import torch.nn.functional as F

import latent_preview
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import node_helpers

GRID_TYPE = "ELEMENT_TILE_GRID"

# All helpers/constants live in common.py and are shared with
# nodes_split.py / nodes_merge.py. This file only holds the node itself.
from .common import (
    LATENT_SCALE,
    _CONTEXT_EDGE,
    _DETAIL_RADIUS,
    _parse_strength_schedule,
    _strip_owned_patches,
    _pixel_alignment,
    _reference_method,
    _tokenizer_handles_images,
    _is_flux2_family,
    _is_krea2,
    _in_context_model,
    _scheduled_ref_model,
    _grounded_conditioning,
    _grounded_reference_conditioning,
    _resolve_geometry,
    _plan_tiles,
    _tile_weight,
    _detail_weight,
    _split_frequencies,
    _match_channels,
    _sanitize_latent,
)


def _tile_sampling_callback(previewer, steps, state):
    """
    Per-tile sampling callback, modelled directly on ComfyUI's own
    latent_preview.prepare_callback (the exact pattern every core sampler node
    uses): one progress bar per tile covering that tile's steps, advanced every
    step together with a fresh latent preview.
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
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", GRID_TYPE)
    RETURN_NAMES = ("image", "split_images", "resampler_images", "grid")
    OUTPUT_IS_LIST = (False, True, True, False)
    FUNCTION = "refine"
    DESCRIPTION = ("All-in-one tiled refine/upscale for reference-latent edit models (Qwen Image "
                   "2.1 / Edit 2509 / 2511, FLUX.2 klein, Krea 2 and similar). Upscales, splits "
                   "into overlapping tiles, regenerates each tile at full tile resolution and "
                   "blends everything into one image. Works with any model/conditioning/vae "
                   "combination that is valid on its own: models without reference-latent support "
                   "(SDXL, Qwen-Image-2512, ...) simply ignore the per-tile reference and run as "
                   "tiled img2img, where feature preservation is governed by denoise alone.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "clip": ("CLIP", ),
                "upscale_by": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "tile_size": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 16,
                    "tooltip": "Resolution each tile is regenerated at. 0 picks one from the output size: "
                               "about four tiles along the longest side, kept between 1024 and 1536."
                               "keep each tile ≤ 2MP for identity-edit LoRAs."}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 16,
                    "tooltip": "How much neighbouring tiles share. 0 = auto: 15% of the tile size, capped "
                               "at 256px (whichever is smaller). Any non-zero value is used as-is, with no "
                               "cap. Higher = smoother seams but more compute."}),
                "feather": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "reference_mode": ("BOOLEAN", {"default": False,
                    "label_on": "Krea 2 identity-edit", "label_off": "conditioning",
                    "tooltip": "Off: tile is attached as reference latent; with 'clip'+'prompt' it also "
                               "enters the text encoder's vision slot. On: Krea 2 identity-edit "
                               "in-context forward, requires 'clip'."}),
                "redo_first_tile": ("BOOLEAN", {"default": True,
                    "label_on": "redo first tile", "label_off": "single pass",
                    "tooltip": "After all other tiles finish, regenerate the first tile once more so it "
                               "can see its finished neighbours as context. Only applies when there is "
                               "more than one tile."}),
                "grounding_px": ("INT", {"default": 768, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Vision-slot size for the per-tile text encode. Qwen Edit 2509/2511: area "
                               "bucket, 0 or >=384 = official 384x384 resize. Qwen Image 2.1: ignored. "
                               "Krea 2: caps the vision input's longest side, 0 = tile size."}),
                "reference_strength": ("STRING", {"default": "1.0,1.0,0.95,0.9,0.8", "multiline": False, "tooltip":
                    "A list of reference strengths for each step, corresponding strictly to the number of steps."
                    "Supports delimiters such as commas (Chinese or English), enumeration commas (、), and colons. A single value applies to all steps; "
                    "if fewer values ​​are provided than the number of steps, the last value is repeated; if more are provided, only the initial portion is used."}),

                "prompt": ("STRING", {"default": "enhance quality preserving all the details",
                    "multiline": True,
                    "tooltip": "Positive condition; encoded tile-by-tile alongside the tile map; cannot be empty."}),

                "negative_prompt": ("STRING", {"default": "", "multiline": True,
                                               "tooltip": "Leave blank = Encode an empty string as the baseline negative condition."}),
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

    def _sample_tile(self, model, sampling_model, positive, negative, vae, tile_pixels, seed, steps, cfg, sampler_name, scheduler, denoise, reference_method=None, source_holder=None, previewer=None, state=None, disable_pbar=True, attach_reference=True):
        """Regenerate one tile: encode it, hand it to the model as reference, sample, decode.
        静态像素放松已移除：encode 进模型的永远是原始干净 tile；
        逐步削弱由模型上的调度（_krea2_in_context_forward / _scheduled_ref_model）在前向内完成。"""
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
            ref_list = [tile_latent]
            positive_tile = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_list}, append=True)
            negative_tile = negative
            if reference_method is not None:
                positive_tile = node_helpers.conditioning_set_values(
                    positive_tile, {"reference_latents_method": reference_method})
    
        latent = tile_latent.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        noise = comfy.sample.prepare_noise(latent, seed, None)
        samples = comfy.sample.sample(sampling_model, noise, steps, cfg, sampler_name, scheduler,
                                      positive_tile, negative_tile, latent, denoise=denoise,
                                      disable_pbar=disable_pbar, seed=seed,
                                      callback=_tile_sampling_callback(previewer, steps, state))
        samples = _sanitize_latent(samples)
        decoded = vae.decode(samples)
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        return decoded


    
    def refine(self, model, vae, image, upscale_by, tile_size, overlap, feather, seed, steps, cfg, sampler_name, scheduler, denoise, reference_mode=False, redo_first_tile=True, grounding_px=768, clip=None, prompt="enhance quality preserving all the details", negative_prompt="", reference_strength="1.0,1.0,0.95,0.9,0.8"):
        if clip is None:
            raise ValueError(
                "TiledUpscale: 'clip' is not connected. The positive/negative input ports have been removed."
                "conditioning is encoded from the clip and the prompt/negative_prompt.")
        if not prompt or not prompt.strip():
            raise ValueError(
                "TiledUpscale: 'prompt' is empty. After the port is removed, the prompt becomes the sole source of positive conditions and cannot be empty.")
        if negative_prompt is None:
            negative_prompt = ""
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        alignment = _pixel_alignment(model)
        strength_values = _parse_strength_schedule(reference_strength)  # 解析每步强度列表

        if reference_mode and not _is_krea2(model.get_model_object("diffusion_model")):
            logging.warning("[TiledUpscale] reference_mode (Krea 2 identity-edit) only applies "
                            "to Krea 2; this model already consumes each tile as a native "
                            "reference latent — falling back to the conditioning path.")
            reference_mode = False
        if reference_mode:
            model = _strip_owned_patches(model)  # schedule + in-context wrapper + extra_conds 全摘
        else:
            model = _strip_owned_patches(model, keys=("tiled_upscale_strength_schedule",), strip_extra_conds=False)
       
        if reference_mode:
            mode = "in_context"
            grounded_conditioning = False
            if clip is None:
                raise ValueError("TiledUpscale: reference_mode (in-context, Krea 2 identity-edit) "
                                 "requires the 'clip' input.")
            source_holder = {"latent": None}
            sampling_model = _in_context_model(model, source_holder, strength_values)   # ← 改：调度进 in-context 前向
            reference_method = None
        else:
            mode = "conditioning"
            grounded_conditioning = False
            source_holder = None
            sampling_model = _scheduled_ref_model(model, strength_values)   # ← 改：恒 1.0 时内部原样返回
            reference_method = _reference_method(model, positive)
            if clip is not None and prompt and _tokenizer_handles_images(clip) and not _is_flux2_family(model):
                grounded_conditioning = True
                mode = "conditioning (grounded)"
            elif clip is not None and prompt:
                dm_name = type(model.get_model_object("diffusion_model")).__name__.lower()
                if "flux3" in dm_name:
                    logging.info("[TiledUpscale] FLUX.3's DiT taps Qwen3-VL text-only (no vision "
                                 "tokens); each tile rides as a native reference latent instead.")
                else:
                    logging.warning("[TiledUpscale] this model's text encoder is text-only (e.g. FLUX.2 "
                                    "klein's Qwen3); the tile cannot enter a vision slot, falling back to "
                                    "the reference-latent-only path with the original conditioning.")
    
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
        logging.info("[TiledUpscale] branch=%s | tokenizer=%s | vision=%s | %d tile(s) | strength schedule=%s.",
                     mode, type(getattr(clip, "tokenizer", None)).__name__ if clip is not None else "none",
                     _tokenizer_handles_images(clip) if clip is not None else False,
                     len(tiles), strength_values)                              
    
        device, dtype = source.device, source.dtype
        channels = source.shape[-1]
        canvas = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
        weight_sum = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)
        detail = torch.zeros_like(canvas)
        detail_sum = torch.zeros_like(weight_sum)
        weight_max = torch.zeros_like(weight_sum)
        context = torch.zeros_like(canvas)
        for info in tiles:
            region = (slice(None), slice(info["y"], info["y"] + info["height"]),
                      slice(info["x"], info["x"] + info["width"]), slice(None))
            weight_max[region] = torch.maximum(weight_max[region], _tile_weight(info, feather, device, dtype))
        detail_radius = max(1, round(overlap * _DETAIL_RADIUS))
    
        def merged_image():
            return (canvas / weight_sum.clamp(min=1e-8)) + (detail / detail_sum.clamp(min=1e-8))
    
        redo_first = len(tiles) > 1 and redo_first_tile
        previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
        state = {"preview": None}
        grounded_in_context = mode == "in_context"
        tile_positive = tile_negative = None
        qwen21 = False
        attach_reference = True
        if grounded_conditioning:
            tok_name = type(getattr(clip, "tokenizer", None)).__name__.lower().replace("_", "")
            qwen21 = ("qwenimage21" in tok_name) and alignment == 32
        global_ref_latent = None
        global_ref_pixels = None
        encode_total = len(tiles) if (grounded_in_context or grounded_conditioning) else 0
        total_runs = encode_total + len(tiles) + (1 if redo_first else 0)
        total_pbar = comfy.utils.ProgressBar(total_runs)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    
        if grounded_in_context or grounded_conditioning:
            tile_positive, tile_negative = [], []
            for encode_index, info in enumerate(tiles):
                total_pbar.update_absolute(encode_index, total_runs, state["preview"])
                logging.info("[TiledUpscale] encoding tile %d/%d (row=%d col=%d) at (%d,%d)",
                             encode_index + 1, encode_total, info["row"], info["col"], info["x"], info["y"])
                crop = source[:, info["y"]:info["y"] + info["height"], info["x"]:info["x"] + info["width"], :]
                if grounded_in_context:
                    tile_positive.append(_grounded_conditioning(clip, prompt, crop, grounding_px))              
                    if cfg > 1.0:
                        tile_negative.append(_grounded_conditioning(clip, negative_prompt, crop, grounding_px)) 
                else:
                    pos_c, neg_c = _grounded_reference_conditioning(
                        clip, prompt, vae, crop, grounding_px, reference_method,
                        encode_negative=(cfg > 1.0), qwen21=qwen21,
                        negative_prompt=negative_prompt)                                
                    tile_positive.append(pos_c)
                    if cfg > 1.0:
                        tile_negative.append(neg_c)
                total_pbar.update_absolute(encode_index + 1, total_runs, state["preview"])
            if grounded_conditioning:
                attach_reference = False
    
        split_images = []
        for info in tiles:
            split_images.append(source[:, info["y"]:info["y"] + info["height"], info["x"]:info["x"] + info["width"], :].clone())
        tiles_out = [None] * len(tiles)
    
        def place_tile(info, previous=None):
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
            
            noise_seed = (seed + info["tile_id"] * 0x9E3779B97F4A7C15) % (1 << 64)
            generated = self._sample_tile(
                model, sampling_model,
                tile_positive[info["tile_id"]] if tile_positive else positive,
                tile_negative[info["tile_id"]] if tile_negative else negative,
                vae, tile_input, noise_seed, steps, cfg, sampler_name, scheduler, denoise,
                reference_method, source_holder, previewer, state, disable_pbar, attach_reference)

            if generated.shape[1] != h or generated.shape[2] != w:
                generated = F.interpolate(generated.permute(0, 3, 1, 2), size=(h, w),
                                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
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
            total_pbar.update_absolute(encode_total + index, total_runs, state["preview"])
            logging.info("[TiledUpscale] sample tile %d/%d (row=%d col=%d) at (%d,%d) size %dx%d",
                         index + 1, len(tiles), info["row"], info["col"], info["x"], info["y"],
                         info["width"], info["height"])
            generated = place_tile(info)
            total_pbar.update_absolute(encode_total + index + 1, total_runs, state["preview"])
            if index == 0 and redo_first:
                first_tile = generated
        if redo_first:
            total_pbar.update_absolute(encode_total + len(tiles), total_runs, state["preview"])
            info = tiles[0]
            logging.info("[TiledUpscale] resample tile 1/1 (row=%d col=%d) at (%d,%d) size %dx%d "
                         "(redo with finished neighbours)", info["row"], info["col"], info["x"], info["y"],
                         info["width"], info["height"])
            place_tile(tiles[0], previous=first_tile)
        total_pbar.update_absolute(total_runs, total_runs, state["preview"])
        merged = merged_image()[:, :img_h, :img_w, :]
        grid = {
            "tiles": tiles, "num_x": num_x, "num_y": num_y,
            "img_h": img_h, "img_w": img_w, "work_h": work_h, "work_w": work_w,
            "overlap": overlap, "tile_size": tile_size, "alignment": alignment,
            "channels": channels, "reference_mode": reference_mode,
        }
        return (merged.clamp(0.0, 1.0), split_images, tiles_out, grid)
    
    
    