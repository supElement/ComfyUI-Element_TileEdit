# nodes_merge.py
import torch
import torch.nn.functional as F
from .common import (_DETAIL_RADIUS, _tile_weight, _match_channels,
                     _split_frequencies, _detail_weight)

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
        w_edge = torch.ones_like(g[..., :1])
    else:
        mask_3d = (weight > 0.01).float().expand_as(g)
        if mask_3d.sum() < 500:
            mask_3d = torch.ones_like(g)
        w_edge = 1.0 - weight
        
    sum_mask = mask_3d.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    
    g_mean = (g * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    r_mean = (r * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    
    g_var = ((g - g_mean)**2 * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    r_var = ((r - r_mean)**2 * mask_3d).sum(dim=(1, 2), keepdim=True) / sum_mask
    
    g_std = g_var.sqrt().clamp_min(0.05)
    r_std = r_var.sqrt().clamp_min(0.05)
    
    scale = (r_std / g_std).clamp(0.5, 2.0)
    matched = (g - g_mean) * scale + r_mean
    
    diff = (matched - g).clamp(-0.35, 0.35)
    
    # 空间加权：中心应用 50% 强度，边缘应用 100% 强度
    if weight is not None:
        # 此时 weight 在中心是 1，边缘是 0
        apply_w = strength * (weight * 0.8 + w_edge * 1.0)
    else:
        apply_w = strength
        
    out = g + diff * apply_w

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
    CATEGORY = "Element TileEdit"
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
                "color_match_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "tile_sources": ("IMAGE", {"tooltip": "..."}),
                "original_image": ("IMAGE", {"tooltip": "..."}),
                "high_low_freq": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable high-low frequency fusion: use feathered weights to smooth the transition for low frequencies, "
                               " and competitive weights to preserve details for high frequencies—preventing details in overlapping regions from being flattened out through averaging."}),
            },
        }

    def merge(self, tile_images, grid, feather, color_match_strength=0.4,
              tile_sources=None, original_image=None, high_low_freq=False):
        grid = _unwrap(grid)
        feather = float(_unwrap(feather, 1.0))
        high_low_freq = bool(_unwrap(high_low_freq, False)) 
        cm_strength = float(_unwrap(color_match_strength, 0.4))
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
        if high_low_freq:
            bg_low_canvas  = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
            bg_low_weight  = torch.zeros(1, work_h, work_w, 1,        device=device, dtype=dtype)
            bg_high_canvas = torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype)
            bg_high_weight = torch.zeros(1, work_h, work_w, 1,        device=device, dtype=dtype)
        
        # === 高低频：按 layer 分组预计算 weight_max（背景一组，每个手绘 layer 各一组）===
        weight_max_bg = None
        weight_max_layers = None
        detail_radius = 0
        if high_low_freq:
            weight_max_bg = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)
            weight_max_layers = {}
            for info_wm in tiles:
                is_fg_wm = info_wm.get("kind") not in {"gap", "fallback", "unified"}
                info_wm_copy = dict(info_wm)
                if overlap > 0:
                    if info_wm_copy.get("fade_top", 0) > 0:    info_wm_copy["fade_top"]    = overlap
                    if info_wm_copy.get("fade_bottom", 0) > 0: info_wm_copy["fade_bottom"] = overlap
                    if info_wm_copy.get("fade_left", 0) > 0:   info_wm_copy["fade_left"]   = overlap
                    if info_wm_copy.get("fade_right", 0) > 0:  info_wm_copy["fade_right"]  = overlap
                w_full = _tile_weight(info_wm_copy, feather, device, dtype)
                xw, yw = info_wm["x"], info_wm["y"]
                ww, hw = info_wm["width"], info_wm["height"]
                x0, y0 = max(0, xw), max(0, yw)
                x1, y1 = min(work_w, xw + ww), min(work_h, yw + hw)
                if x1 <= x0 or y1 <= y0:
                    continue
                w_part = w_full[:, y0 - yw:y1 - yw, x0 - xw:x1 - xw, :]
                region_wm = (slice(None), slice(y0, y1), slice(x0, x1), slice(None))
                if is_fg_wm:
                    layer_idx_wm = info_wm.get("layer", 0)
                    if layer_idx_wm is None: layer_idx_wm = 0
                    if layer_idx_wm not in weight_max_layers:
                        weight_max_layers[layer_idx_wm] = torch.zeros(1, work_h, work_w, 1, device=device, dtype=dtype)
                    wm = weight_max_layers[layer_idx_wm]
                    wm[region_wm] = torch.maximum(wm[region_wm], w_part)
                else:
                    weight_max_bg[region_wm] = torch.maximum(weight_max_bg[region_wm], w_part)
            detail_radius = max(1, round(overlap * _DETAIL_RADIUS))

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

            # 3. 获取 split 传过来的像素级 Mask
            is_fg = info.get("kind") not in {"gap", "fallback", "unified"}
            tile_mask = info.get("mask")
            if tile_mask is not None:
                tile_mask = tile_mask.to(device=device, dtype=dtype)
                if tile_mask.ndim == 3:
                    tile_mask = tile_mask.unsqueeze(-1)

            # 4. 计算 Pad 大小并执行拉伸 (保留原流程)
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
                
                # 【新增】Mask 也一起 pad，用 replicate 保持边缘的形状
                if tile_mask is not None:
                    tile_mask = F.pad(tile_mask.permute(0, 3, 1, 2), (pad_l, pad_r, pad_t, pad_b), mode='replicate').permute(0, 2, 3, 1)
            weight_raw = weight
            
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
                decoded_c    = decoded[:, dy_start:dy_end, dx_start:dx_end, :]
                weight_c     = weight[:, dy_start:dy_end, dx_start:dx_end, :]
                weight_raw_c = weight_raw[:, dy_start:dy_end, dx_start:dx_end, :]   
                tile_mask_c  = (tile_mask[:, dy_start:dy_end, dx_start:dx_end, :]
                                if tile_mask is not None else None)                 
                region = (slice(None), slice(ty_start, ty_end), slice(tx_start, tx_end), slice(None))
            else:
                decoded_c    = decoded
                weight_c     = weight
                weight_raw_c = weight_raw
                tile_mask_c  = tile_mask
                region = (slice(None), slice(0, 0), slice(0, 0), slice(None))

            # 7. 颜色匹配
            if cm_strength > 0.0 and src is not None:
                if tx_start < tx_end and ty_start < ty_end:
                    src_c = src[:, dy_start:dy_end, dx_start:dx_end, :]
                else:
                    src_c = src
                decoded_c = _color_match(decoded_c, src_c, cm_strength, weight_c)

            # 8. 累加到画布（高低频 / 原逻辑）
            has_region = (tx_start < tx_end and ty_start < ty_end)
            
            if high_low_freq:
                if has_region:
                    low, high = _split_frequencies(decoded_c, detail_radius)
            
                    # 按 layer 取 weight_max：手绘层只跟同层兄弟竞争，不被背景压制
                    if is_fg:
                        layer_idx_wm = info.get("layer", 0)
                        if layer_idx_wm is None: layer_idx_wm = 0
                        wm_src = weight_max_layers.get(layer_idx_wm, weight_max_bg)
                    else:
                        wm_src = weight_max_bg
                    weight_max_c = wm_src[region]
            
                    # 竞争口径：weight_raw（不含 mask）对 weight_max（不含 mask）→ 同口径
                    dw = _detail_weight(weight_raw_c, weight_max_c)
            
                    dw_eff = dw * tile_mask_c if tile_mask_c is not None else dw
            
                    if is_fg:
                        layer_idx = info.get("layer", 0)
                        if layer_idx is None: layer_idx = 0
                        if layer_idx not in layer_groups:
                            layer_groups[layer_idx] = (
                                torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype),  # low canvas
                                torch.zeros(1, work_h, work_w, 1,        device=device, dtype=dtype),  # low weight
                                torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype),  # high canvas
                                torch.zeros(1, work_h, work_w, 1,        device=device, dtype=dtype),  # high weight
                            )
                        lc, lw_, hc, hw_ = layer_groups[layer_idx]
                        lc[region] += low  * weight_c
                        lw_[region] += weight_c
                        hc[region] += high * dw_eff
                        hw_[region] += dw_eff
                    else:
                        bg_low_canvas[region]  += low  * weight_c
                        bg_low_weight[region]  += weight_c
                        bg_high_canvas[region] += high * dw_eff
                        bg_high_weight[region] += dw_eff
                # has_region=False：region 为空，跳过
            else:
                if is_fg:
                    layer_idx = info.get("layer", 0)
                    if layer_idx is None: layer_idx = 0
                    if layer_idx not in layer_groups:
                        layer_groups[layer_idx] = (
                            torch.zeros(1, work_h, work_w, channels, device=device, dtype=dtype),
                            torch.zeros(1, work_h, work_w, 1,        device=device, dtype=dtype),
                        )
                    lc, lw_ = layer_groups[layer_idx]
                    lc[region] += decoded_c * weight_c
                    lw_[region] += weight_c
                else:
                    bg_canvas[region] += decoded_c * weight_c
                    bg_weight[region] += weight_c

        if high_low_freq:
            bg_img   = (bg_low_canvas  / bg_low_weight.clamp(min=1e-8)) \
                     + (bg_high_canvas / bg_high_weight.clamp(min=1e-8))
            bg_alpha = bg_low_weight.clamp(0.0, 1.0)
        else:
            bg_img   = bg_canvas / bg_weight.clamp(min=1e-8)
            bg_alpha = bg_weight.clamp(0.0, 1.0)
        
        if base_canvas is not None:
            base_merged = bg_img * bg_alpha + base_canvas * (1.0 - bg_alpha)
        else:
            base_merged = bg_img

        # === 手绘图层按 layer_index 从大到小逐层 alpha over，最上方图层最后拼接===
        merged = base_merged
        for layer_idx in sorted(layer_groups.keys(), reverse=True):
            if high_low_freq:
                lc, lw, hc, hw = layer_groups[layer_idx]
                layer_img   = (lc / lw.clamp(min=1e-8)) + (hc / hw.clamp(min=1e-8))
                layer_alpha = lw.clamp(0.0, 1.0)
            else:
                lc, lw = layer_groups[layer_idx]
                layer_img   = lc / lw.clamp(min=1e-8)
                layer_alpha = lw.clamp(0.0, 1.0)
            merged = layer_img * layer_alpha + merged * (1.0 - layer_alpha)

        img_h, img_w = grid["img_h"], grid["img_w"]
        return (merged[:, :img_h, :img_w, :].clamp(0.0, 1.0),)

