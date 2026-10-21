# nodes_split.py
"""ElementTileSplit — 纯几何分块 + 图像源（零模型依赖）。

- 图像来源：image 输入端口（优先）> 前端拖拽/粘贴上传的图（记录在 layout_data.source）
- 三种模式：manual（手画框图层）/ count / size（统一网格，忽略手画框）
- 未覆盖区域智能补块（fill_gaps，贪心最大矩形）
- 每层 upscale（生成分辨率）+ c×r 二次分割 + 层内 overlap
- 提示词合成：layer + ", " + global（层为空则仅 global）
- 输出：tile_images（源 crop 重采样到生成尺寸）、per-tile prompt 列表、grid（给 Merge/Encode）

坐标系约定：
- 框/层坐标 = 源图像素空间（前端画布所见即所得，吸附 div）
- region/tile 坐标 = 输出画布空间（源 × unified_upscale，ceil 对齐 div），merge 直接贴回
- gen（生成分辨率）= 源区域 × 该层 upscale，div 取整（VAE 合法尺寸，可超采样）
"""

import json
import logging
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image, ImageOps

import comfy.utils
import folder_paths

GRID_TYPE = "ELEMENT_TILE_GRID"

PREVIEW_MAX_EDGE = 2048   # 执行后节点内预览长边上限（仅预览，不影响输出）
MAX_TILES = 256

# ================================================================
# Gap 规划 v4：core 禁区 + B&B 精确最少覆盖 + 块数目标驱动的自动加深覆盖
# ================================================================
GAP_MAX_ASPECT     = 3.0   # 块长宽比上限（违例会触发下一档更深覆盖）
GAP_MIN_SHORT_DIVS = 2     # 块最短边下限（× div）
GAP_COVER_STEPS    = (0, 8, 16, 25)  # 允许覆盖手绘框短边的比例（%）
GAP_TARGET_TILES   = 8     # 块数达标线（想更少就调小，如 4~6）
GAP_SEARCH_BUDGET  = 1200  # 每轮 B&B 节点预算

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _snap(v, div):
    """四舍五入到 div 的整数倍。"""
    return int(round(v / div)) * div


def _snap_ceil(v, div):
    """向上取整到 div 的整数倍（画布只放大不裁切，保证边缘不被切掉）。"""
    return max(div, int(math.ceil(v / div - 1e-6)) * div)


def _round_div(v, div):
    return max(div, _snap(v, div))


def _compose(layer_p, global_p, append_mode=True):
    """根据模式组合提示词。"""
    l_clean = str(layer_p).strip() if layer_p else ""
    g_clean = str(global_p).strip() if global_p else ""
    
    if append_mode:
        parts = []
        if l_clean: parts.append(l_clean)
        if g_clean: parts.append(g_clean)
        return ", ".join(parts)
    else:
        return l_clean if l_clean else g_clean


def _parse_state(raw):
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _norm_layers(data):
    """前端 JSON 里的手画框图层 → 后端结构化列表（数组顺序 = 图层顺序）。"""
    layers = []
    for i, l in enumerate(data.get("layers") or []):
        if not isinstance(l, dict):
            continue
        try:
            x, y = int(round(float(l.get("x", 0)))), int(round(float(l.get("y", 0))))
            w, h = int(round(float(l.get("w", 0)))), int(round(float(l.get("h", 0))))
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        try:
            up = _clamp(float(l.get("upscale", 2.0) or 2.0), 0.1, 16.0)
        except Exception:
            up = 2.0
        layers.append({
            "id": l.get("id", i),
            "color": str(l.get("color", "#888888")),
            "x": x, "y": y, "w": w, "h": h,
            "upscale": up,
            "cols": _clamp(int(l.get("cols", 1) or 1), 1, 16),
            "rows": _clamp(int(l.get("rows", 1) or 1), 1, 16),
            "prompt": str(l.get("prompt", "") or ""),
            "negative": str(l.get("negative", "") or ""),
        })
    return layers


def _overlap_px(mode, px_val, pct_val, tile_edge):
    """by px / by % 开关式二选一；% 基准 = 该层 tile 边长（输出空间）。"""
    if mode == "pct":
        return int(round(_clamp(float(pct_val), 0.0, 45.0) / 100.0 * tile_edge))
    return _clamp(int(px_val or 0), 0, max(0, int(tile_edge) // 2))

def _inject_manual_overlap(tiles, default_ov, out_w, out_h):
    """对所有块（手绘框和背景块）强制向外扩展 default_ov/2。
    因为每个块都往外扩了，所以边界必定产生物理重叠，Merge 节点才能正确羽化。"""
    if default_ov <= 0: return
    half = max(0, default_ov // 2)
    if half == 0: return
    
    for t in tiles:
        # 无论 layer 还是 gap，统一向外扩展，并限制在画布内
        nx = max(0, t["x"] - half)
        ny = max(0, t["y"] - half)
        nw = min(out_w - nx, t["width"] + (t["x"] - nx) + half)
        nh = min(out_h - ny, t["height"] + (t["y"] - ny) + half)
        
        t["x"] = nx
        t["y"] = ny
        t["width"] = nw
        t["height"] = nh

def _compute_manual_fades(tiles, default_ov, out_w, out_h):
    """基于 tile 之间的真实 overlap 计算 fade_top/bottom/left/right。"""
    n = len(tiles)
    half = max(1, default_ov // 2)
    
    for t in tiles:
        t.setdefault("fade_top", 0)
        t.setdefault("fade_bottom", 0)
        t.setdefault("fade_left", 0)
        t.setdefault("fade_right", 0)

    # 1. 处理背景块之间的羽化
    for i in range(n):
        for j in range(i + 1, n):
            t, other = tiles[i], tiles[j]
            
            # 如果其中一个是手绘框，直接跳过，因为我们后面会强制固定它
            if t.get("kind") == "layer" or other.get("kind") == "layer":
                continue
                
            x_over = min(t["x"] + t["width"], other["x"] + other["width"]) - max(t["x"], other["x"])
            y_over = min(t["y"] + t["height"], other["y"] + other["height"]) - max(t["y"], other["y"])
            if x_over <= 0 or y_over <= 0:
                continue
            
            fade_val_x = min(x_over, default_ov)
            fade_val_y = min(y_over, default_ov)

            tcx = t["x"] + t["width"] / 2
            ocx = other["x"] + other["width"] / 2
            if ocx > tcx:
                t["fade_right"] = max(t["fade_right"], fade_val_x)
                other["fade_left"] = max(other["fade_left"], fade_val_x)
            else:
                t["fade_left"] = max(t["fade_left"], fade_val_x)
                other["fade_right"] = max(other["fade_right"], fade_val_x)

            tcy = t["y"] + t["height"] / 2
            ocy = other["y"] + other["height"] / 2
            if ocy > tcy:
                t["fade_bottom"] = max(t["fade_bottom"], fade_val_y)
                other["fade_top"] = max(other["fade_top"], fade_val_y)
            else:
                t["fade_top"] = max(t["fade_top"], fade_val_y)
                other["fade_bottom"] = max(other["fade_bottom"], fade_val_y)

    # 2. 强制手绘框的羽化四边绝对相等，并且等于外扩量
    for t in tiles:
        if t.get("kind") == "layer":
            # 限制单边 fade 不超过宽高的 45%，防止框太小导致中心没有实心
            fw = min(half, t["width"] * 0.45)
            fh = min(half, t["height"] * 0.45)
            # 只有在画布内部的边才赋予羽化，贴着画布边缘的边不需要羽化
            t["fade_top"] = fh if t["y"] > 0 else 0
            t["fade_bottom"] = fh if t["y"] + t["height"] < out_h else 0
            t["fade_left"] = fw if t["x"] > 0 else 0
            t["fade_right"] = fw if t["x"] + t["width"] < out_w else 0

    # 3. 全局绝对保护：所有块（无论 layer 还是 gap）的贴边处，羽化必须强制为 0
    for t in tiles:
        if t["x"] <= 0:
            t["fade_left"] = 0
        if t["y"] <= 0:
            t["fade_top"] = 0
        if t["x"] + t["width"] >= out_w:
            t["fade_right"] = 0
        if t["y"] + t["height"] >= out_h:
            t["fade_bottom"] = 0

# ---------------------------------------------------------------- 几何切分

def _split_region(x, y, w, h, cols, rows, ov):
    """把一个 region 切成 cols×rows 个带 overlap 的子块，完整覆盖 [x,x+w]×[y,y+h]。
    块宽 = (w + ov*(cols-1)) / cols（相邻块重叠 ov），末块贴齐右/下边界。"""
    cols, rows, ov = max(1, int(cols)), max(1, int(rows)), max(0, int(ov))
    if cols == 1 and rows == 1:
        return [{"x": x, "y": y, "w": w, "h": h, "col": 0, "row": 0,
                 "fade_top": 0, "fade_bottom": 0, "fade_left": 0, "fade_right": 0}]

    def _axis(total, n, o):
        if n == 1:
            return [(0, total)]
        cw = max(1, int(math.floor((total + o * (n - 1)) / n + 0.5)))
        step = (total - cw) / (n - 1)
        out = []
        for i in range(n):
            a = int(math.floor(i * step + 0.5))
            end = a + cw if i < n - 1 else total
            out.append((a, end - a))
        return out

    xs = _axis(w, cols, ov)
    ys = _axis(h, rows, ov)
    out = []
    for j, (ty, th) in enumerate(ys):
        fade_top = max(0, (ys[j-1][0] + ys[j-1][1]) - ty) if j > 0 else 0
        fade_bottom = max(0, (ty + th) - ys[j+1][0]) if j < len(ys) - 1 else 0
        for i, (tx, tw) in enumerate(xs):
            fade_left = max(0, (xs[i-1][0] + xs[i-1][1]) - tx) if i > 0 else 0
            fade_right = max(0, (tx + tw) - xs[i+1][0]) if i < len(xs) - 1 else 0
            out.append({
                "x": x + tx, "y": y + ty, "w": tw, "h": th, "col": i, "row": j,
                "fade_top": fade_top, "fade_bottom": fade_bottom,
                "fade_left": fade_left, "fade_right": fade_right
            })
    return out

def _find_largest_free_rect(free):
    """free[r][c]=True 为空闲。逐行直方图法求最大全空闲矩形。"""
    rows, cols = len(free), len(free[0])
    heights = [0] * cols
    balanced = None   # (area, r0, c0, h, w) 纵横比合格的最大矩形
    widest   = None   # (area, r0, c0, h, w) 兜底：面积最大（可能窄）
    for r in range(rows):
        for c in range(cols):
            heights[c] = heights[c] + 1 if free[r][c] else 0
        stack = []
        c = 0
        while c <= cols:
            hcur = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= hcur:
                s, hgt = stack.pop()
                area = hgt * (c - s)
                if area > 0:
                    cand = (area, r - hgt + 1, s, hgt, c - s)
                    if widest is None or area > widest[0]:
                        widest = cand
                    w_ = c - s
                    aspect = max(hgt, w_) / max(1, min(hgt, w_))
                    if balanced is None or area > balanced[0]:
                        balanced = cand
                start = s
            stack.append((start, hcur))
            c += 1
    pick = widest
    if not pick:
        return None
    return pick[1], pick[2], pick[3], pick[4]



def _snap(v, step):
    return int(round(float(v) / step)) * step


def _balanced_subsplit(rects, gap_max_px):
    """超 gap_max 的块做内部均衡切分（选子块长宽比最接近 1 的方案）。"""
    out = []
    for (rx, ry, rw, rh) in rects:
        if rw <= gap_max_px and rh <= gap_max_px:
            out.append((rx, ry, rw, rh)); continue
        max_sc = max(1, -(-rw // gap_max_px)); max_sr = max(1, -(-rh // gap_max_px))
        best = None
        for sc in range(1, max_sc + 1):
            sw = -(-rw // sc)
            if sw > gap_max_px: continue
            for sr in range(1, max_sr + 1):
                sh = -(-rh // sr)
                if sh > gap_max_px: continue
                a = max(sw, sh) / max(1, min(sw, sh))
                if best is None or a < best[0]: best = (a, sc, sr)
        sc, sr = (max_sc, max_sr) if best is None else (best[1], best[2])
        sw, sh = -(-rw // sc), -(-rh // sr)
        for i in range(sc):
            for j in range(sr):
                nx, ny = rx + i * sw, ry + j * sh
                out.append((nx, ny, min(sw, rx + rw - nx), min(sh, ry + rh - ny)))
    return out


def _merge_adjacent(rects, gap_max_px, min_short, aspect=GAP_MAX_ASPECT):
    """相邻且并集仍为矩形、不超限的块两两合并（消解切分碎片）。
    aspect 可放宽合并后的长宽比上限（v6 用 5.0 以吸收切出来的长条）。"""
    rs = [list(r) for r in rects]
    changed = True
    while changed:
        changed = False
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if a[1] == b[1] and a[3] == b[3] and (a[0] + a[2] == b[0] or b[0] + b[2] == a[0]):
                    m = [min(a[0], b[0]), a[1], a[2] + b[2], a[3]]
                elif a[0] == b[0] and a[2] == b[2] and (a[1] + a[3] == b[1] or b[1] + b[3] == a[1]):
                    m = [a[0], min(a[1], b[1]), a[2], a[3] + b[3]]
                else:
                    continue
                if (m[2] > gap_max_px or m[3] > gap_max_px or
                        max(m[2], m[3]) / max(1, min(m[2], m[3])) > aspect or
                        min(m[2], m[3]) < min_short):
                    continue
                rs[i] = m; rs.pop(j); changed = True
                break
            if changed: break
    return [tuple(r) for r in rs]

def _rect_overlap(a, b):
    """两个 (x, y, w, h) 矩形的重叠面积。"""
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return w * h

def _rect_subtract(a, b):
    """返回 a 减去 b 之后的矩形列表（a、b 均为 (x, y, w, h) 整数元组）。"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 >= ix2 or iy1 >= iy2:
        return [a]
    if ix1 <= ax and iy1 <= ay and ix2 >= ax + aw and iy2 >= ay + ah:
        return []
    out = []
    if iy1 > ay:
        out.append((ax, ay, aw, iy1 - ay))
    if iy2 < ay + ah:
        out.append((ax, iy2, aw, ay + ah - iy2))
    if ix1 > ax:
        out.append((ax, iy1, ix1 - ax, iy2 - iy1))
    if ix2 < ax + aw:
        out.append((ix2, iy1, ax + aw - ix2, iy2 - iy1))
    return out

def _edge_fully_covered(B, R):
    """判断矩形 R 是否完整跨越了矩形 B 的某一条边。"""
    bx, by, bw, bh = B
    rx, ry, rw, rh = R
    # 必须有交集
    ix1, iy1 = max(bx, rx), max(by, ry)
    ix2, iy2 = min(bx + bw, rx + rw), min(by + bh, ry + rh)
    if ix1 >= ix2 or iy1 >= iy2:
        return False
    
    # 上边被完全覆盖
    if rx <= bx and rx + rw >= bx + bw and ry <= by and ry + rh > by:
        return True
    # 下边被完全覆盖
    if rx <= bx and rx + rw >= bx + bw and ry < by + bh and ry + rh >= by + bh:
        return True
    # 左边被完全覆盖
    if ry <= by and ry + rh >= by + bh and rx <= bx and rx + rw > bx:
        return True
    # 右边被完全覆盖
    if ry <= by and ry + rh >= by + bh and rx < bx + bw and rx + rw >= bx + bw:
        return True
    return False

def _plan_gap_tiles(covered, div, out_w, out_h, gap_max_px, overlap_px=0,
                    regions=None, target_tiles=GAP_TARGET_TILES):
    """【严格按照需求】先均分、扣除手绘框、长条变胖合并、超尺寸平分。"""
    if regions is None:
        regions = []
        
    # 1. 先不考虑手绘框，即先均分
    base_grid = _balanced_subsplit([(0, 0, out_w, out_h)], gap_max_px)
    
    # 2. 考虑手绘框，只有边被完全覆盖才切割（否则保留整块，允许盖住框）
    pieces = []
    for b in base_grid:
        parts = [b]
        for rg in regions:
            rr = (rg["x"], rg["y"], rg["w"], rg["h"])
            nxt = []
            for p in parts:
                if _edge_fully_covered(p, rr):
                    nxt.extend(_rect_subtract(p, rr))
                else:
                    nxt.append(p) # 没有完全覆盖边，直接保留整块
            parts = nxt
            if not parts: break
        pieces.extend(parts)
        
    # 3. 若切割后出现比例大于5：1的长条合并到邻居分块（向四周扩展吃框，遇到邻居就合并）
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            if pieces[i] is None: continue
            x, y, w, h = pieces[i]
            if max(w, h) / max(1, min(w, h)) <= 5.0:
                continue
                
            ox, oy, ow, oh = x, y, w, h
            # 长条向外扩展（稍微覆盖手绘框），直到比例降至 5:1
            if w > h:
                diff = w / 5.0 - h
                expand_t = min(y, diff / 2)
                expand_b = diff - expand_t
                if expand_b > out_h - (y + h): 
                    expand_b = out_h - (y + h)
                    expand_t = diff - expand_b
                y = max(0, int(y - expand_t))
                h = min(out_h - y, int(h + diff))
            else:
                diff = h / 5.0 - w
                expand_l = min(x, diff / 2)
                expand_r = diff - expand_l
                if expand_r > out_w - (x + w):
                    expand_r = out_w - (x + w)
                    expand_l = diff - expand_r
                x = max(0, int(x - expand_l))
                w = min(out_w - x, int(w + diff))
                
            # 扩展后的矩形，如果与其他块重叠，就合并到邻居分块（取外接矩形）
            nx, ny, nw, nh = x, y, w, h
            for j in range(len(pieces)):
                if i == j or pieces[j] is None: continue
                q = pieces[j]
                ix = max(nx, q[0]); iy = max(ny, q[1])
                ix2 = min(nx + nw, q[0] + q[2]); iy2 = min(ny + nh, q[1] + q[3])
                if ix < ix2 and iy < iy2:
                    nx2 = min(nx, q[0]); ny2 = min(ny, q[1])
                    nx = nx2; ny = ny2
                    nw = max(nx + nw, q[0] + q[2]) - nx
                    nh = max(ny + nh, q[1] + q[3]) - ny
                    pieces[j] = None 
            
            # 只有尺寸发生变化才更新，防止贴边时无限循环
            if (nx, ny, nw, nh) != (ox, oy, ow, oh):
                pieces[i] = (nx, ny, nw, nh)
                changed = True
            
        if changed:
            pieces = [p for p in pieces if p is not None]
            
    # 4. 若合并后超过max side参数，则对这个分块做平分
    final_pieces = []
    for p in pieces:
        if p[2] > gap_max_px or p[3] > gap_max_px:
            final_pieces.extend(_balanced_subsplit([p], gap_max_px))
        else:
            final_pieces.append(p)
            
    # 清理因扩展合并可能产生的微调重叠
    clean_pieces = []
    for p in final_pieces:
        parts = [p]
        for c in clean_pieces:
            nxt = []
            for pt in parts:
                nxt.extend(_rect_subtract(pt, c))
            parts = nxt
            if not parts: break
        clean_pieces.extend(parts)
        
    # 应用 overlap 向外扩展（修正边界）
    if overlap_px > 0 and clean_pieces:
        half = max(0, overlap_px // 2)
        res = []
        for (x, y, w, h) in clean_pieces:
            nx = max(0, x - half)
            ny = max(0, y - half)
            nw = min(out_w - nx, x + w + half - nx)
            nh = min(out_h - ny, y + h + half - ny)
            res.append((nx, ny, nw, nh))
        clean_pieces = res
        
    return clean_pieces


# ---------------------------------------------------------------- 图像源 / 预览

def _load_source_image(data, image):
    """优先级：image 端口 > layout_data.source（拖拽/粘贴上传图）。"""
    if image is not None:
        img = image
        if img.ndim == 2:
            img = img.unsqueeze(0)
        if img.ndim == 3:
            img = img.unsqueeze(0)
        if img.shape[0] > 1:
            logging.warning("[TileSplit] batch of %d connected; using the first frame",
                            img.shape[0])
        img = img[:1]
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img.contiguous()

    src = data.get("source") or {}
    name = str(src.get("filename", "") or "").strip()
    if not name:
        raise ValueError(
            "ElementTileSplit: no image source. Connect an image input, or drop/paste "
            "an image onto the node in the browser.")
    subfolder = str(src.get("subfolder", "") or "").strip()
    path = folder_paths.get_annotated_filepath(f"{subfolder}/{name}" if subfolder else name)
    if not os.path.exists(path):
        raise ValueError(f"ElementTileSplit: image file not found: {name} "
                         f"(re-drop or re-paste it in the node UI)")
    i = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    arr = np.array(i).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _save_preview(image, unique_id):
    """执行后节点内预览，双写（对齐 Element_ImageCurve 链路）：
    - temp 目录 + 唯一文件名 → ui["bg_image"]，前端 onExecuted 实时显示
    - input 目录 + 固定文件名 → ui["preview"]，持久引用，重启后仍可显示
    """
    try:
        _, h, w, _ = image.shape
        src_meta = {"w": int(w), "h": int(h)}   # ← 新增：源图真实尺寸（注意此时 image 还没被缩小）
        scale = min(1.0, PREVIEW_MAX_EDGE / max(h, w))
        if scale < 1.0:
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            image = comfy.utils.common_upscale(
                image.permute(0, 3, 1, 2), nw, nh, "area", "disabled").permute(0, 2, 3, 1)
        arr = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        bg, ref = None, None
        try:
            temp_dir = folder_paths.get_temp_directory()
            os.makedirs(temp_dir, exist_ok=True)
            name = f"tilesplit_preview_{int(time.time() * 1000)}_{random.randint(1, 1000000)}.png"
            pil.save(os.path.join(temp_dir, name), format="PNG")
            bg = [{"filename": name, "subfolder": "", "type": "temp", **src_meta}]   # ← 改：追加 **src_meta
        except Exception as e:
            logging.warning("[TileSplit] preview temp failed: %s", e)
        try:
            sub = "element_tile_split/preview"
            out_dir = os.path.join(folder_paths.get_input_directory(), sub)
            os.makedirs(out_dir, exist_ok=True)
            name = f"ets_preview_{unique_id}.jpg"
            pil.save(os.path.join(out_dir, name), format="JPEG", quality=92)
            ref = {"filename": name, "subfolder": sub, "type": "input", **src_meta}  # ← 改：追加 **src_meta
        except Exception as e:
            logging.warning("[TileSplit] preview persist failed: %s", e)
        return bg, ref
    except Exception as e:
        logging.warning("[TileSplit] preview save failed: %s", e)
        return None, None


# ---------------------------------------------------------------- 节点

class ElementTileSplit:
    """纯几何分块节点：画布/图层 UI 状态经 layout_data(JSON widget) 传入，
    所有几何在执行时后端全量重算校验（前端仅为预览，防陈旧数据）。"""

    CATEGORY = "Element TileUpscale"
    FUNCTION = "split"
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", GRID_TYPE)
    RETURN_NAMES = ("split_images", "original_image", "prompt", "negative_prompt", "grid")
    OUTPUT_IS_LIST = (True, False, True, True, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "layout_data": ("STRING", {"default": "{}", "multiline": True,
                                           "tooltip": "Internal UI state (hidden)."}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def split(self, layout_data="{}", image=None, unique_id=None):
        data = _parse_state(layout_data)
        div = _clamp(int(data.get("div", 16) or 16), 2, 256)
        mode = str(data.get("mode", "manual") or "manual")
        unified = data.get("unified") or {}
        try:
            u_up = _clamp(float(unified.get("upscale", 2.0) or 2.0), 0.1, 16.0)
        except Exception:
            u_up = 2.0
        u_cols = _clamp(int(unified.get("columns", 2) or 2), 1, 16)
        u_rows = _clamp(int(unified.get("rows", 2) or 2), 1, 16)
        u_tile = _clamp(int(unified.get("tile_size", 1024) or 1024), div, 8192)
        ov_mode = str(data.get("overlap_mode", "px") or "px")
        ov_px = data.get("overlap_px", 64)
        ov_pct = data.get("overlap_pct", 15)
        fill_gaps = bool(data.get("fill_gaps", True))
        gap_max = _clamp(int(data.get("gap_max_px", 1536) or 1536), div, 8192)
        g_prompt = str(data.get("global_prompt", "") or "")
        g_neg = str(data.get("global_negative", "") or "")
        g_append = bool(data.get("global_append", True))
        enable_lup = bool(data.get("enable_layer_upscale", False))
        layers = _norm_layers(data)

        source = _load_source_image(data, image)
        _, src_h, src_w, channels = source.shape

        # 输出画布 = 源 × unified，ceil 对齐 div（只放大不裁切）
        out_w = _snap_ceil(src_w * u_up, div)
        out_h = _snap_ceil(src_h * u_up, div)

        tiles, prompts, negatives, ov_values = [], [], [], []

        def emit(subs, layer=None, layer_index=-1):
            for t in subs:
                up_val = u_up
                if layer is not None and enable_lup:
                    up_val = layer["upscale"]

                if layer is not None:
                    p = _compose(layer["prompt"], g_prompt, g_append)
                    n = _compose(layer["negative"], g_neg, g_append)
                else:
                    p, n = g_prompt.strip(), g_neg.strip()
                prompts.append(p)
                negatives.append(n)
                tiles.append({
                    "x": t["x"], "y": t["y"], "width": t["w"], "height": t["h"],
                    "col": t.get("col", 0), "row": t.get("row", 0),
                    "layer": layer_index, "layer_id": layer["id"] if layer else None,
                    "upscale": up_val,  
                    "kind": t.get("kind", "layer"),
                    "fade_top": t.get("fade_top", 0),
                    "fade_bottom": t.get("fade_bottom", 0),
                    "fade_left": t.get("fade_left", 0),
                    "fade_right": t.get("fade_right", 0),
                })

        manual_mode = mode == "manual"
        covered = None

        if manual_mode:
            # 手画框 → 输出空间 region（div 吸附；贴边的框延伸到画布边缘）
            layer_regions = []
            for l in layers:
                rx = _clamp(_snap(l["x"] * u_up, div), 0, out_w - div)
                ry = _clamp(_snap(l["y"] * u_up, div), 0, out_h - div)
                if l["x"] + l["w"] >= src_w - 0.5:
                    rw = out_w - rx
                else:
                    rw = _clamp(_snap(l["w"] * u_up, div), div, out_w - rx)
                if l["y"] + l["h"] >= src_h - 0.5:
                    rh = out_h - ry
                else:
                    rh = _clamp(_snap(l["h"] * u_up, div), div, out_h - ry)
                layer_regions.append((l, {"x": rx, "y": ry, "w": rw, "h": rh}))

            # covered 网格：有 fill_gaps 就分配（无框时 = 整幅画布都视为未覆盖）
            if fill_gaps:
                cols_n, rows_n = out_w // div, out_h // div
                covered = [[False] * cols_n for _ in range(rows_n)]
                for _, rg in layer_regions:
                    c0, r0 = rg["x"] // div, rg["y"] // div
                    c1 = (rg["x"] + rg["w"] + div - 1) // div
                    r1 = (rg["y"] + rg["h"] + div - 1) // div
                    for r in range(max(0, r0), min(r1, rows_n)):
                        row = covered[r]
                        for c in range(max(0, c0), min(c1, cols_n)):
                            row[c] = True

            for li, (l, rg) in enumerate(layer_regions):
                if l["cols"] > 1 or l["rows"] > 1:
                    edge = min(rg["w"] / l["cols"], rg["h"] / l["rows"])
                    ov = _overlap_px(ov_mode, ov_px, ov_pct, edge)
                    ov_values.append(ov)
                else:
                    ov = 0
                subs = _split_region(rg["x"], rg["y"], rg["w"], rg["h"],
                                     l["cols"], l["rows"], ov)
                for s in subs:
                    s["kind"] = "layer"
                emit(subs, layer=l, layer_index=li)

            if not layer_regions and not fill_gaps:
                logging.warning("[TileSplit] manual mode with no boxes; "
                                "whole canvas as one tile")
                emit([{"x": 0, "y": 0, "w": out_w, "h": out_h,
                       "col": 0, "row": 0, "kind": "fallback"}])

            # 完全尊重 UI 传过来的 Overlap 设置，不再进行不合理的钳制
            if ov_mode == "px":
                default_ov = max(0, int(ov_px or 0))
            else:
                # 对于百分比模式，基于整个画布的短边计算像素
                default_ov = int(round(min(out_w, out_h) * (_clamp(float(ov_pct), 0.0, 45.0) / 100.0)))

            if fill_gaps and covered is not None:
                # 提取纯坐标列表传进去
                rg_list = [rg for _, rg in layer_regions]
                # 传入 overlap_px=0，因为我们在最后会对所有 tiles 统一外扩
                for i, (gx, gy, gw, gh) in enumerate(
                        _plan_gap_tiles(covered, div, out_w, out_h, gap_max,
                                        overlap_px=0, regions=rg_list)):
                    emit([{"x": gx, "y": gy, "w": gw, "h": gh,
                           "col": i, "row": 0, "kind": "gap"}])
        else:
            # 统一模式：by count 用 columns×rows；by size 按 tile_size（输出 px）摊
            if mode == "size":
                cols = max(1, -(-out_w // u_tile))
                rows = max(1, -(-out_h // u_tile))
            else:
                cols, rows = u_cols, u_rows
            ov = _overlap_px(ov_mode, ov_px, ov_pct, min(out_w / cols, out_h / rows))
            ov_values.append(ov)
            subs = _split_region(0, 0, out_w, out_h, cols, rows, ov)
            for s in subs:
                s["kind"] = "unified"
            emit(subs)


        if not tiles:
            logging.warning("[TileSplit] no tiles from state (mode=%s, layers=%d, "
                            "fill_gaps=%s); falling back to whole-canvas single tile",
                            mode, len(layers), fill_gaps)
            emit([{"x": 0, "y": 0, "w": out_w, "h": out_h,
                   "col": 0, "row": 0, "kind": "fallback"}])

        if len(tiles) > MAX_TILES:
            raise ValueError(f"ElementTileSplit: {len(tiles)} tiles exceed MAX_TILES={MAX_TILES}.")

        # ---- manual 模式收尾：为所有 tile 计算 fade 值 + 网格 overlap ----
        if manual_mode:
            # default_ov = ov_values[0] if ov_values else min(64, div * 4)
            # 层与层 / 层与间隙 / 间隙与间隙 之间注入 overlap（仅对无 overlap 的边生效）
            _inject_manual_overlap(tiles, default_ov, out_w, out_h)  # 加上 out_w 和 out_h
            # 根据 tile 之间的真实 overlap，计算每个 tile 的 fade_top/bottom/left/right
            _compute_manual_fades(tiles, default_ov, out_w, out_h)
            # 网格 overlap = 所有 tile 间实际存在的最大 overlap
            grid_overlap = max(
                (max(t.get("fade_top", 0), t.get("fade_bottom", 0),
                     t.get("fade_left", 0), t.get("fade_right", 0))
                 for t in tiles),
                default=0,
            )
            grid_overlap = max(grid_overlap, max(ov_values) if ov_values else 0)
        else:
            grid_overlap = max(ov_values) if ov_values else 0

        # 每 tile：源空间 crop（= region / unified）→ 重采样到 gen（= 源区域 × 该层 upscale）
        pbar = comfy.utils.ProgressBar(len(tiles))
        tile_images, grid_tiles = [], []
        for idx, t in enumerate(tiles):
            # 增加防御：确保坐标和尺寸不会超出画布边界
            tx = max(0, min(t["x"], out_w - 1))
            ty = max(0, min(t["y"], out_h - 1))
            tw = min(t["width"], out_w - tx)
            th = min(t["height"], out_h - ty)

            up = t["upscale"]
            cx = _clamp(int(round(tx / u_up)), 0, src_w - 1)
            cy = _clamp(int(round(ty / u_up)), 0, src_h - 1)
            cw = _clamp(int(round(tw / u_up)), 1, src_w - cx)
            ch = _clamp(int(round(th / u_up)), 1, src_h - cy)
            crop = source[:, cy:cy + ch, cx:cx + cw, :].contiguous()

            gen_w = _clamp(_round_div(cw * up, div), div, 8192)
            gen_h = _clamp(_round_div(ch * up, div), div, 8192)
            if (gen_w, gen_h) != (cw, ch):
                method = "lanczos" if gen_w * gen_h >= cw * ch else "area"
                crop = comfy.utils.common_upscale(
                    crop.permute(0, 3, 1, 2), gen_w, gen_h, method, "disabled"
                ).permute(0, 2, 3, 1).contiguous()
            tile_images.append(crop)
            grid_tiles.append({
                "x": tx, "y": ty, "width": tw, "height": th,
                "row": t["row"], "col": t["col"],
                "gen_w": gen_w, "gen_h": gen_h,
                "layer": t["layer"], "layer_id": t["layer_id"], "kind": t["kind"],
                "fade_top": t.get("fade_top", 0),
                "fade_bottom": t.get("fade_bottom", 0),
                "fade_left": t.get("fade_left", 0),
                "fade_right": t.get("fade_right", 0),
                "weight_mult": t.get("weight_mult", 1.0), 
            })
            pbar.update_absolute(idx + 1, len(tiles))

        grid = {
            "tiles": grid_tiles,
            "num_x": 0, "num_y": 0,          # 非均匀网格，无全局列/行
            "img_h": out_h, "img_w": out_w,
            "work_h": out_h, "work_w": out_w,
            "overlap": grid_overlap,
            "tile_size": 0,
            "alignment": div,
            "channels": int(channels),
            "mode": mode,
            "unified_upscale": u_up,
            "div": div,
        }

        bg, pref = _save_preview(source, unique_id)
        ui = {"src": {"w": int(src_w), "h": int(src_h)}, "batch_size": [1]}
        if bg:
            ui["bg_image"] = bg
        if pref:
            ui["preview"] = pref


        logging.info("[TileSplit] mode=%s | %d tile(s) | canvas %dx%d (div=%d, unified=%.2fx)",
                     mode, len(tiles), out_w, out_h, div, u_up)
        return {"ui": ui, "result": (tile_images, source, prompts, negatives, grid)}

