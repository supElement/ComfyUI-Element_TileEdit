# nodes_merge.py
import torch
import torch.nn.functional as F

from .common import (_DETAIL_RADIUS, _tile_weight, _detail_weight,
                     _split_frequencies, _match_channels)

GRID_TYPE = "ELEMENT_TILE_GRID"


def _unwrap(value, default=None):
    """INPUT_IS_LIST=True 时 ComfyUI 会把每个输入都包成一层 list；取第一个。"""
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if len(value) > 0 else default
    return value


def _color_match(generated, reference, strength=1.0):
    """逐通道 mean/std 色彩迁移（Ultimate SD Upscale 'Color fix: Mean+Std' 同类实现）。
    strength=0 直接原样返回（= 关闭）；>0 时把 generated 的每通道均值/方差
    按 strength 比例向 reference 对齐。只搬统计量、不动结构：色调/曝光向原图分块
    靠拢，重采样带来的新细节全部保留。"""
    if strength <= 0.0:
        return generated
    g = generated.float()
    r = reference.float()
    g_mean = g.mean(dim=(1, 2), keepdim=True)          # (1,1,1,C)
    g_std = g.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    r_mean = r.mean(dim=(1, 2), keepdim=True)
    r_std = r.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    matched = (g - g_mean) / g_std * r_std + r_mean
    out = matched * strength + g * (1.0 - strength)    # 中间值 = 部分保留原色调
    return out.clamp(0.0, 1.0).to(generated.dtype)


class ElementTileMerge:
    """Merge per-tile DECODED images back into one image using the same low/high
    frequency separation + feather weighting. This node does NOT run the VAE:
    decode the tile_latents with ComfyUI's native 'VAE Decode' (or 'VAE Decode
    (Tiled)') first, then plug its IMAGE list output in here."""
    CATEGORY = "Element TileUpscale"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "merge"
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_images": ("IMAGE",),
                "grid": (GRID_TYPE,),
                "feather": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "tile_sources": ("IMAGE", {"tooltip": "Split nodes divide the original image into blocks for output，"
                                                       "As a reference for color matching"}),
                "color_match_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,
                                                    "step": 0.05,
                                                    "tooltip": "0 = Disabled. >0: Performs channel-wise mean/std color transfer"
                                                               "between each resampled block and the corresponding block in the original image "
                                                               "(adjusting tone while preserving detail)."}),
            },
        }

    def merge(self, tile_images, grid, feather, tile_sources=None, color_match_strength=0.0):
        grid = _unwrap(grid)
        feather = float(_unwrap(feather, 1.0))
        cm_strength = float(_unwrap(color_match_strength, 0.0))

        tiles = grid["tiles"]
        work_h, work_w = grid["work_h"], grid["work_w"]
        channels = grid["channels"]
        overlap = grid["overlap"]

        if len(tile_images) < len(tiles):
            raise ValueError(f"ElementTileMerge: got {len(tile_images)} decoded image(s) "
                             f"for {len(tiles)} grid tile(s).")
        if tile_sources is not None and len(tile_sources) < len(tiles):
            raise ValueError("ElementTileMerge: tile_sources count doesn't match the grid.")
        if cm_strength > 0.0 and tile_sources is None:
            raise ValueError("ElementTileMerge: color_match_strength > 0 connection required "
                             "tile_sources（Split nodes divide the original image into blocks for output）。")

        first = _unwrap(tile_images)
        device, dtype = first.device, first.dtype

        canvas = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
        weight_sum = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)
        detail = torch.zeros_like(canvas)
        detail_sum = torch.zeros_like(weight_sum)
        weight_max = torch.zeros_like(weight_sum)

        for info in tiles:
            region = (slice(None),
                      slice(info["y"], info["y"] + info["height"]),
                      slice(info["x"], info["x"] + info["width"]),
                      slice(None))
            weight_max[region] = torch.maximum(
                weight_max[region], _tile_weight(info, feather, device, dtype))

        detail_radius = max(1, round(overlap * _DETAIL_RADIUS))

        for idx, info in enumerate(tiles):
            decoded = tile_images[idx]
            if decoded.ndim == 3:
                decoded = decoded.unsqueeze(0)
            decoded = decoded[:1].to(device=device, dtype=dtype)
            decoded = _match_channels(decoded, channels)

            x, y, w, h = info["x"], info["y"], info["width"], info["height"]

            # 色彩匹配：0 = 关闭（连参考图处理都跳过）
            if cm_strength > 0.0:
                src = tile_sources[idx].to(device=device, dtype=dtype)
                if src.ndim == 3:
                    src = src.unsqueeze(0)
                decoded = _color_match(decoded, src[:1], cm_strength)

            if decoded.shape[1] != h or decoded.shape[2] != w:
                decoded = F.interpolate(decoded.permute(0, 3, 1, 2), size=(h, w),
                                        mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            region = (slice(None), slice(y, y + h), slice(x, x + w), slice(None))
            weight = _tile_weight(info, feather, device, dtype)
            dw = _detail_weight(weight, weight_max[region])

            low, high = _split_frequencies(decoded, detail_radius)
            canvas[region] += low * weight
            weight_sum[region] += weight
            detail[region] += high * dw
            detail_sum[region] += dw

        merged = (canvas / weight_sum.clamp(min=1e-8)) + (detail / detail_sum.clamp(min=1e-8))
        img_h, img_w = grid["img_h"], grid["img_w"]
        return (merged[:, :img_h, :img_w, :].clamp(0.0, 1.0),)
