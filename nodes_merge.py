# nodes_merge.py
import torch
import torch.nn.functional as F
from .common import (_DETAIL_RADIUS, _tile_weight, _match_channels)

GRID_TYPE = "ELEMENT_TILE_GRID"

def _unwrap(value, default=None):
    if value is None: return default
    if isinstance(value, list): return value[0] if len(value) > 0 else default
    return value

def _gaussian_blur_isotropic(x, sigma_px):
    """各向同性可分离高斯模糊。x: (B, C, H, W)，返回同 shape。"""
    if sigma_px < 0.5:
        return x
    radius = max(1, int(round(sigma_px * 3.0)))   # 覆盖 99.7% 高斯能量
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma_px ** 2))
    kernel = kernel / kernel.sum()
    c = x.shape[1]
    kw = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1).contiguous()
    kh = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1).contiguous()
    x = F.conv2d(x, kw, padding=(0, radius), groups=c)
    x = F.conv2d(x, kh, padding=(radius, 0), groups=c)
    return x

def _color_match(generated, reference, strength=1.0, weight=None):
    if strength <= 0.0: return generated
    
    g = generated.float()
    r = reference.float()

    # --- 防止传进来的参考图尺寸不同 ---
    if g.shape[1:3] != r.shape[1:3]:
        r = F.interpolate(r.permute(0, 3, 1, 2), size=(g.shape[1], g.shape[2]), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    if weight is not None and (weight.shape[1] != g.shape[1] or weight.shape[2] != g.shape[2]):
        weight = F.interpolate(weight.permute(0, 3, 1, 2), size=(g.shape[1], g.shape[2]), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    # ------------------------------

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
    
    g_std = g_var.sqrt().clamp_min(0.05)
    r_std = r_var.sqrt().clamp_min(0.05)
    
    matched = (g - g_mean) / g_std * r_std + r_mean

    out = matched * strength + g * (1.0 - strength)
    return out.clamp(0.0, 1.0).to(generated.dtype)

def _generate_tile_mask(w, h, rtl, rtr, rbl, rbr, device, dtype):
    """根据当前分块的尺寸，直接生成尺寸为 (w, h) 的硬边缘 Mask。
    【核心】直接在当前外扩后的坐标系中切出圆角，完美保留外扩后的圆弧。"""
    ys = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    # 初始全为 1
    mask = torch.ones_like(xx, dtype=torch.bool)

    # 直接以当前分块的外框为基准切圆角
    if rtl > 0:
        m = (xx < rtl) & (yy < rtl)
        dist = torch.sqrt((xx - rtl)**2 + (yy - rtl)**2)
        mask &= ~(m & (dist > rtl))
    if rtr > 0:
        m = (xx > w - rtr) & (yy < rtr)
        dist = torch.sqrt((xx - (w - rtr))**2 + (yy - rtr)**2)
        mask &= ~(m & (dist > rtr))
    if rbl > 0:
        m = (xx < rbl) & (yy > h - rbl)
        dist = torch.sqrt((xx - rbl)**2 + (yy - (h - rbl))**2)
        mask &= ~(m & (dist > rbl))
    if rbr > 0:
        m = (xx > w - rbr) & (yy > h - rbr)
        dist = torch.sqrt((xx - (w - rbr))**2 + (yy - (h - rbr))**2)
        mask &= ~(m & (dist > rbr))

    return mask.float().unsqueeze(0).unsqueeze(-1).to(dtype)


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

        bg_canvas = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
        bg_weight = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)

        # === 手绘 layer 按图层分组 ===
        layer_groups = {}
        alpha_cache = {}

        triples = []
        for i in range(len(tiles)):
            info = tiles[i]
            decoded = tile_images[i]
            src = None
            if cm_strength > 0.0 and tile_sources is not None and i < len(tile_sources):
                src = tile_sources[i]
            triples.append((info, decoded, src))

        for info, decoded, src in triples:
            if decoded.ndim == 3: decoded = decoded.unsqueeze(0)
            decoded = decoded[:1].to(device=device, dtype=dtype)
            decoded = _match_channels(decoded, channels)

            # 1. 获取【原始尺寸】并缩放图像
            x, y, w, h = info["x"], info["y"], info["width"], info["height"]
            if decoded.shape[1] != h or decoded.shape[2] != w:
                decoded = F.interpolate(decoded.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
            
            if cm_strength > 0.0 and src is not None:
                if src.ndim == 3: src = src.unsqueeze(0)
                src = src[:1].to(device=device, dtype=dtype)
                if src.shape[1] != h or src.shape[2] != w:
                    src = F.interpolate(src.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)

            # 2. 计算基础权重
            # 【修改】只对内部重叠的边(fade > 0)强制拉满到 overlap(96px)，贴边的边(fade = 0)坚决不动
            if overlap > 0:
                info = dict(info)
                if info.get("fade_top", 0) > 0:    info["fade_top"] = overlap
                if info.get("fade_bottom", 0) > 0: info["fade_bottom"] = overlap
                if info.get("fade_left", 0) > 0:   info["fade_left"] = overlap
                if info.get("fade_right", 0) > 0:  info["fade_right"] = overlap
                
            weight = _tile_weight(info, feather, device, dtype)

            # 3. 获取硬 Mask (如果是手绘框且有圆角)
            is_fg = info.get("kind") not in {"gap", "fallback", "unified"}
            tile_mask = None
            if is_fg:
                rtl = int(info.get("radius_tl", 0))
                rtr = int(info.get("radius_tr", 0))
                rbl = int(info.get("radius_bl", 0))
                rbr = int(info.get("radius_br", 0))
                if rtl or rtr or rbl or rbr:
                    tile_mask = _generate_tile_mask(
                        w, h, rtl, rtr, rbl, rbr, device, dtype
                    )

            # 4. 计算 Pad 大小并执行拉伸
            overlap = grid.get("overlap", 0)
            pad_l = overlap if x <= 0 else 0
            pad_r = overlap if x + w >= work_w else 0
            pad_t = overlap if y <= 0 else 0
            pad_b = overlap if y + h >= work_h else 0
            
            px_start = x - pad_l
            px_end = x + w + pad_r
            py_start = y - pad_t
            py_end = y + h + pad_b

            if pad_l > 0 or pad_r > 0 or pad_t > 0 or pad_b > 0:
                decoded = F.pad(decoded.permute(0, 3, 1, 2), (pad_l, pad_r, pad_t, pad_b), mode='replicate').permute(0, 2, 3, 1)
                if src is not None:
                    src = F.pad(src.permute(0, 3, 1, 2), (pad_l, pad_r, pad_t, pad_b), mode='replicate').permute(0, 2, 3, 1)
                
                weight = F.pad(weight.permute(0, 3, 1, 2), (pad_l, pad_r, pad_t, pad_b), mode='replicate').permute(0, 2, 3, 1)
                if tile_mask is not None:
                    tile_mask = F.pad(tile_mask.permute(0, 3, 1, 2), (pad_l, pad_r, pad_t, pad_b), mode='replicate').permute(0, 2, 3, 1)

            # 5. 【Pad 之后】对硬 Mask 进行羽化模糊
            if tile_mask is not None:
                max_fade = max(info.get("fade_top", 0), info.get("fade_bottom", 0), info.get("fade_left", 0), info.get("fade_right", 0))
                feather_px = max(1.0, max_fade * float(feather)) if max_fade > 0 else max(1.0, overlap * float(feather))
                
                if feather_px > 1.0:
                    sigma = feather_px / 3.0
                    tm = tile_mask.permute(0, 3, 1, 2)
                    tm = _gaussian_blur_isotropic(tm, sigma)
                    tile_mask = tm.permute(0, 2, 3, 1)
                    
                weight = weight * tile_mask

            # 6. 切掉画布外的部分
            dx_start = max(0, -px_start)
            dy_start = max(0, -py_start)
            dx_end = decoded.shape[2] - max(0, px_end - work_w)
            dy_end = decoded.shape[1] - max(0, py_end - work_h)
            
            tx_start = max(0, px_start)
            tx_end = min(work_w, px_end)
            ty_start = max(0, py_start)
            ty_end = min(work_h, py_end)

            if tx_start < tx_end and ty_start < ty_end:
                decoded_c = decoded[:, dy_start:dy_end, dx_start:dx_end, :]
                weight_c = weight[:, dy_start:dy_end, dx_start:dx_end, :]
                region = (slice(None), slice(ty_start, ty_end), slice(tx_start, tx_end), slice(None))
            else:
                decoded_c = decoded
                weight_c = weight
                region = (slice(None), slice(0, 0), slice(0, 0), slice(None))

            # 7. 颜色匹配
            if cm_strength > 0.0 and src is not None:
                if tx_start < tx_end and ty_start < ty_end:
                    src_c = src[:, dy_start:dy_end, dx_start:dx_end, :]
                else:
                    src_c = src
                decoded_c = _color_match(decoded_c, src_c, cm_strength, weight_c)

            # 8. 累加到画布
            if is_fg:
                layer_idx = info.get("layer", 0)
                if layer_idx is None: layer_idx = 0
                if layer_idx not in layer_groups:
                    layer_groups[layer_idx] = (
                        torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype),
                        torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype),
                    )
                lc, lw_ = layer_groups[layer_idx]
                lc[region] += decoded_c * weight_c
                lw_[region] += weight_c
            else:
                bg_canvas[region] += decoded_c * weight_c
                bg_weight[region] += weight_c

        bg_img = bg_canvas / bg_weight.clamp(min=1e-8)
        bg_alpha = bg_weight.clamp(0.0, 1.0)
        if base_canvas is not None:
            base_merged = bg_img * bg_alpha + base_canvas * (1.0 - bg_alpha)
        else:
            base_merged = bg_img

        # === 手绘图层按 layer_index 从大到小逐层 alpha over，最上方图层最后拼接===
        merged = base_merged
        for layer_idx in sorted(layer_groups.keys(), reverse=True):
            lc, lw = layer_groups[layer_idx]
            layer_img = lc / lw.clamp(min=1e-8)
            layer_alpha = lw.clamp(0.0, 1.0)
            merged = layer_img * layer_alpha + merged * (1.0 - layer_alpha)

        img_h, img_w = grid["img_h"], grid["img_w"]
        return (merged[:, :img_h, :img_w, :].clamp(0.0, 1.0),)

