# nodes_merge.py
import torch
import torch.nn.functional as F
from .common import (_DETAIL_RADIUS, _tile_weight, _match_channels)

GRID_TYPE = "ELEMENT_TILE_GRID"

def _unwrap(value, default=None):
    if value is None: return default
    if isinstance(value, list): return value[0] if len(value) > 0 else default
    return value

def _color_match(generated, reference, strength=1.0, weight=None):
    if strength <= 0.0: return generated
    
    g = generated.float()
    r = reference.float()
    
    if weight is None:
        mask_3d = torch.ones_like(g)
    else:
        edge_mask = (weight < 0.8).float()
        if edge_mask.sum() < 500:
            edge_mask = torch.ones_like(weight)
        mask_3d = edge_mask.expand_as(g)
        
    sum_mask = mask_3d.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    
    g_mean = (g * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    r_mean = (r * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    
    g_var = ((g - g_mean)**2 * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    r_var = ((r - r_mean)**2 * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    
    g_std = g_var.sqrt().clamp_min(1e-6)
    r_std = r_var.sqrt().clamp_min(1e-6)
    
    matched = (g - g_mean) / g_std * r_std + r_mean

    out = matched * strength + g * (1.0 - strength)
    return out.clamp(0.0, 1.0).to(generated.dtype)

class ElementTileMerge:
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
                "tile_sources": ("IMAGE", {"tooltip": "..."}),
                "original_image": ("IMAGE", {"tooltip": "..."}),
                "color_match_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def merge(self, tile_images, grid, feather, tile_sources=None, original_image=None, color_match_strength=0.0):
        grid = _unwrap(grid)
        feather = float(_unwrap(feather, 1.0))
        cm_strength = float(_unwrap(color_match_strength, 0.0))
        tiles = grid["tiles"]
        work_h, work_w = grid["work_h"], grid["work_w"]
        channels = grid["channels"]
        overlap = grid["overlap"]

        # 1. 准备原图
        base_canvas = None
        if original_image is not None:
            src = _unwrap(original_image)
            if src.ndim == 3: src = src.unsqueeze(0)
            src = src[:1].to(dtype=torch.float32)
            up = float(grid.get("unified_upscale", 1.0) or 1.0)
            src_bchw = src.permute(0, 3, 1, 2)
            _, _, sh, sw = src_bchw.shape
            target_h = work_h if up >= 1.0 else max(1, round(sh * up))
            target_w = work_w if up >= 1.0 else max(1, round(sw * up))
            if src_bchw.shape[-2:] != (target_h, target_w):
                src_bchw = F.interpolate(src_bchw.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
            if src_bchw.shape[-2:] != (work_h, work_w):
                src_bchw = F.interpolate(src_bchw.float(), size=(work_h, work_w), mode="bilinear", align_corners=False)
            base_canvas = src_bchw.permute(0, 2, 3, 1)

        first = _unwrap(tile_images)
        device, dtype = first.device, first.dtype
        if base_canvas is not None:
            base_canvas = base_canvas.to(device=device, dtype=dtype)

        # 2. 准备累加器 (仅保留低频)
        bg_canvas = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
        bg_weight = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)

        fg_canvas = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
        fg_weight = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)

        triples = []
        for i in range(len(tiles)):
            info = tiles[i]
            decoded = tile_images[i]
            src = None
            if cm_strength > 0.0 and tile_sources is not None and i < len(tile_sources):
                src = tile_sources[i]
            triples.append((info, decoded, src))

        # 3. 直接累加图层像素 (纯 Alpha 加权混合)
        for info, decoded, src in triples:
            if decoded.ndim == 3: decoded = decoded.unsqueeze(0)
            decoded = decoded[:1].to(device=device, dtype=dtype)
            decoded = _match_channels(decoded, channels)

            x, y, w, h = info["x"], info["y"], info["width"], info["height"]

            region = (slice(None), slice(y, y + h), slice(x, x + w), slice(None))
            weight = _tile_weight(info, feather, device, dtype) # 提前计算权重图

            if cm_strength > 0.0 and src is not None:
                if src.ndim == 3: src = src.unsqueeze(0)
                src = src[:1].to(device=device, dtype=dtype)
                # 传入 weight 进行边缘匹配
                decoded = _color_match(decoded, src, cm_strength, weight)

            if decoded.shape[1] != h or decoded.shape[2] != w:
                decoded = F.interpolate(decoded.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            # region = (slice(None), slice(y, y + h), slice(x, x + w), slice(None))
            # weight = _tile_weight(info, feather, device, dtype)
            is_fg = info.get("kind") not in {"gap", "fallback", "unified"}

            if is_fg:
                fg_canvas[region] += decoded * weight
                fg_weight[region] += weight
            else:
                bg_canvas[region] += decoded * weight
                bg_weight[region] += weight

        # 4. 还原为完整的图层图像
        bg_img = bg_canvas / bg_weight.clamp(min=1e-8)
        bg_alpha = bg_weight.clamp(0.0, 1.0)
        
        fg_img = fg_canvas / fg_weight.clamp(min=1e-8)
        fg_alpha = fg_weight.clamp(0.0, 1.0)

        # # 5. 图层叠加合成 (Normal 模式)
        # # 底层：原图打底，叠加背景
        # if base_canvas is not None:
            # base_merged = bg_img * bg_alpha + base_canvas * (1.0 - bg_alpha)
        # else:
            # base_merged = bg_img * bg_alpha

        # 5. 图层叠加合成 (Normal 模式)
        # 底层：原图打底，叠加背景
        if base_canvas is not None:
            base_merged = bg_img * bg_alpha + base_canvas * (1.0 - bg_alpha)
        else:
            # 如果没有连接原图，直接使用分块的归一化颜色，避免因为边缘 alpha<1 导致变黑底
            base_merged = bg_img


        # 顶层：手绘框平滑盖上去
        merged = fg_img * fg_alpha + base_merged * (1.0 - fg_alpha)

        img_h, img_w = grid["img_h"], grid["img_w"]
        return (merged[:, :img_h, :img_w, :].clamp(0.0, 1.0),)