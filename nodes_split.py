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

# 与前端 CORNER_RADIUS_SCALES_WITH_UPSCALE 保持一致
CORNER_RADIUS_SCALES_WITH_UPSCALE = True

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
            "r_tl": max(0, int(l.get("r_tl", 0) or 0)),
            "r_tr": max(0, int(l.get("r_tr", 0) or 0)),
            "r_bl": max(0, int(l.get("r_bl", 0) or 0)),
            "r_br": max(0, int(l.get("r_br", 0) or 0)),
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

    for i in range(n):
        for j in range(i + 1, n):
            t, other = tiles[i], tiles[j]
            
            # 如果其中一个是手绘框，直接跳过，后面会强制固定它
            # if t.get("kind") == "layer" or other.get("kind") == "layer":
                # continue
                
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
            fw = min(half, t["width"] * 0.45)
            fh = min(half, t["height"] * 0.45)
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

# ================================================================
# 圆弧求交 + 单刀切割 + 安全探测扩张（与前端逐位镜像）
# ================================================================

def _get_top_y(lx, R):
    rx, ry, rw, rh = R["x"], R["y"], R["w"], R["h"]
    rtl = R.get("rtl", 0) or 0
    rtr = R.get("rtr", 0) or 0
    if lx < rx: return ry
    if lx > rx + rw: return ry
    if lx < rx + rtl:
        cx, cy = rx + rtl, ry + rtl
        dx = cx - lx
        return cy - math.sqrt(max(0.0, rtl * rtl - dx * dx))
    if lx > rx + rw - rtr:
        cx, cy = rx + rw - rtr, ry + rtr
        dx = lx - cx
        return cy - math.sqrt(max(0.0, rtr * rtr - dx * dx))
    return ry


def _get_bottom_y(lx, R):
    rx, ry, rw, rh = R["x"], R["y"], R["w"], R["h"]
    rbl = R.get("rbl", 0) or 0
    rbr = R.get("rbr", 0) or 0
    if lx < rx: return ry + rh
    if lx > rx + rw: return ry + rh
    if lx < rx + rbl:
        cx, cy = rx + rbl, ry + rh - rbl
        dx = cx - lx
        return cy + math.sqrt(max(0.0, rbl * rbl - dx * dx))
    if lx > rx + rw - rbr:
        cx, cy = rx + rw - rbr, ry + rh - rbr
        dx = lx - cx
        return cy + math.sqrt(max(0.0, rbr * rbr - dx * dx))
    return ry + rh


def _get_left_x(ly, R):
    rx, ry, rw, rh = R["x"], R["y"], R["w"], R["h"]
    rtl = R.get("rtl", 0) or 0
    rbl = R.get("rbl", 0) or 0
    if ly < ry: return rx
    if ly > ry + rh: return rx
    if ly < ry + rtl:
        cx, cy = rx + rtl, ry + rtl
        dy = cy - ly
        return cx - math.sqrt(max(0.0, rtl * rtl - dy * dy))
    if ly > ry + rh - rbl:
        cx, cy = rx + rbl, ry + rh - rbl
        dy = ly - cy
        return cx - math.sqrt(max(0.0, rbl * rbl - dy * dy))
    return rx


def _get_right_x(ly, R):
    rx, ry, rw, rh = R["x"], R["y"], R["w"], R["h"]
    rtr = R.get("rtr", 0) or 0
    rbr = R.get("rbr", 0) or 0
    if ly < ry: return rx + rw
    if ly > ry + rh: return rx + rw
    if ly < ry + rtr:
        cx, cy = rx + rw - rtr, ry + rtr
        dy = cy - ly
        return cx + math.sqrt(max(0.0, rtr * rtr - dy * dy))
    if ly > ry + rh - rbr:
        cx, cy = rx + rw - rbr, ry + rh - rbr
        dy = ly - cy
        return cx + math.sqrt(max(0.0, rbr * rbr - dy * dy))
    return rx + rw


def _subtract_single_cut(p, R):
    """与前端 subtractSingleCut 逐位镜像。None 表示 B 完全被 A 包含。"""
    px, py, pw, ph = p
    pb, pr = py + ph, px + pw

    ox = max(px, R["x"]); oy = max(py, R["y"])
    ox2 = min(pr, R["x"] + R["w"]); oy2 = min(pb, R["y"] + R["h"])
    if ox >= ox2 or oy >= oy2:
        return p

    rtl = R.get("rtl", 0) or 0
    rtr = R.get("rtr", 0) or 0
    rbl = R.get("rbl", 0) or 0
    rbr = R.get("rbr", 0) or 0

    def _inside(x, y):
        if x < R["x"] - 1e-6 or x > R["x"] + R["w"] + 1e-6: return False
        if y < R["y"] - 1e-6 or y > R["y"] + R["h"] + 1e-6: return False
        if rtl > 0 and x < R["x"] + rtl and y < R["y"] + rtl:
            return (x - (R["x"] + rtl))**2 + (y - (R["y"] + rtl))**2 <= rtl*rtl + 1e-4
        if rtr > 0 and x > R["x"] + R["w"] - rtr and y < R["y"] + rtr:
            return (x - (R["x"] + R["w"] - rtr))**2 + (y - (R["y"] + rtr))**2 <= rtr*rtr + 1e-4
        if rbl > 0 and x < R["x"] + rbl and y > R["y"] + R["h"] - rbl:
            return (x - (R["x"] + rbl))**2 + (y - (R["y"] + R["h"] - rbl))**2 <= rbl*rbl + 1e-4
        if rbr > 0 and x > R["x"] + R["w"] - rbr and y > R["y"] + R["h"] - rbr:
            return (x - (R["x"] + R["w"] - rbr))**2 + (y - (R["y"] + R["h"] - rbr))**2 <= rbr*rbr + 1e-4
        return True

    if _inside(px, py) and _inside(pr, py) and _inside(px, pb) and _inside(pr, pb):
        return None

    cov_top    = (R["x"] <= px and R["x"] + R["w"] >= pr and R["y"] <= py and R["y"] + R["h"] > py) and _inside(px, py) and _inside(pr, py)
    cov_bottom = (R["x"] <= px and R["x"] + R["w"] >= pr and R["y"] <  pb and R["y"] + R["h"] >= pb) and _inside(px, pb) and _inside(pr, pb)
    cov_left   = (R["y"] <= py and R["y"] + R["h"] >= pb and R["x"] <= px and R["x"] + R["w"] > px) and _inside(px, py) and _inside(px, pb)
    cov_right  = (R["y"] <= py and R["y"] + R["h"] >= pb and R["x"] <  pr and R["x"] + R["w"] >= pr) and _inside(pr, py) and _inside(pr, pb)

    if not (cov_top or cov_bottom or cov_left or cov_right):
        return p

    best_area = float("inf"); best_type = None; best_cut = 0.0

    if cov_top:
        cut_y = min(_get_bottom_y(px, R), _get_bottom_y(pr, R))
        cut_y = max(float(py), min(float(pb), cut_y))
        area = pw * (pb - cut_y)
        if 0 <= area < best_area: best_area = area; best_type = 'top'; best_cut = cut_y
    if cov_bottom:
        cut_y = max(_get_top_y(px, R), _get_top_y(pr, R))
        cut_y = max(float(py), min(float(pb), cut_y))
        area = pw * (cut_y - py)
        if 0 <= area < best_area: best_area = area; best_type = 'bottom'; best_cut = cut_y
    if cov_left:
        cut_x = min(_get_right_x(py, R), _get_right_x(pb, R))
        cut_x = max(float(px), min(float(pr), cut_x))
        area = ph * (pr - cut_x)
        if 0 <= area < best_area: best_area = area; best_type = 'left'; best_cut = cut_x
    if cov_right:
        cut_x = max(_get_left_x(py, R), _get_left_x(pb, R))
        cut_x = max(float(px), min(float(pr), cut_x))
        area = ph * (cut_x - px)
        if 0 <= area < best_area: best_area = area; best_type = 'right'; best_cut = cut_x

    if best_type is None:
        return p

    nx, ny, nw, nh = float(px), float(py), float(pw), float(ph)
    if best_type == 'top':    ny = best_cut; nh = pb - best_cut
    elif best_type == 'bottom': ny = py;     nh = best_cut - py
    elif best_type == 'left':   nx = best_cut; nw = pr - best_cut
    elif best_type == 'right':  nx = px;       nw = best_cut - px

    if nw <= 0 or nh <= 0:
        return None
    return (nx, ny, nw, nh)


def _rect_intersects_rounded_rect(rect, R):
    rx, ry, rw, rh = rect
    rx2, ry2 = rx + rw, ry + rh

    if rx2 <= R["x"] or rx >= R["x"] + R["w"] or ry2 <= R["y"] or ry >= R["y"] + R["h"]:
        return False

    rtl = R.get("rtl", 0) or 0
    rtr = R.get("rtr", 0) or 0
    rbl = R.get("rbl", 0) or 0
    rbr = R.get("rbr", 0) or 0

    cx0 = R["x"] + max(rtl, rbl)
    cx1 = R["x"] + R["w"] - max(rtr, rbr)
    cy0 = R["y"] + max(rtl, rtr)
    cy1 = R["y"] + R["h"] - max(rbl, rbr)
    if rx < cx1 and rx2 > cx0 and ry < cy1 and ry2 > cy0:
        return True

    def circle_intersects(cx, cy, r):
        nx = max(rx, min(cx, rx2))
        ny = max(ry, min(cy, ry2))
        return (nx - cx) ** 2 + (ny - cy) ** 2 < r * r

    if rtl > 0:
        cx = R["x"] + rtl; cy = R["y"] + rtl
        nx = max(rx, min(cx, rx2)); ny = max(ry, min(cy, ry2))
        if nx <= cx and ny <= cy and circle_intersects(cx, cy, rtl): return True
    if rtr > 0:
        cx = R["x"] + R["w"] - rtr; cy = R["y"] + rtr
        nx = max(rx, min(cx, rx2)); ny = max(ry, min(cy, ry2))
        if nx >= cx and ny <= cy and circle_intersects(cx, cy, rtr): return True
    if rbl > 0:
        cx = R["x"] + rbl; cy = R["y"] + R["h"] - rbl
        nx = max(rx, min(cx, rx2)); ny = max(ry, min(cy, ry2))
        if nx <= cx and ny >= cy and circle_intersects(cx, cy, rbl): return True
    if rbr > 0:
        cx = R["x"] + R["w"] - rbr; cy = R["y"] + R["h"] - rbr
        nx = max(rx, min(cx, rx2)); ny = max(ry, min(cy, ry2))
        if nx >= cx and ny >= cy and circle_intersects(cx, cy, rbr): return True
    return False


def _get_safe_expand(rect, direction, max_dist, regions):
    x, y, w, h = rect
    safe = 0.0
    step = 4.0
    while safe < max_dist:
        d = min(max_dist - safe, step)
        if direction == 'top':      test_rect = (x, y - safe - d, w, d)
        elif direction == 'bottom': test_rect = (x, y + h + safe, w, d)
        elif direction == 'left':   test_rect = (x - safe - d, y, d, h)
        elif direction == 'right':  test_rect = (x + w + safe, y, d, h)
        else: break

        hit = any(_rect_intersects_rounded_rect(test_rect, rg) for rg in regions)
        if hit:
            if d > 1:
                d = 1
                if direction == 'top':      test_rect = (x, y - safe - 1, w, 1)
                elif direction == 'bottom': test_rect = (x, y + h + safe, w, 1)
                elif direction == 'left':   test_rect = (x - safe - 1, y, 1, h)
                elif direction == 'right':  test_rect = (x + w + safe, y, 1, h)
                hit1 = any(_rect_intersects_rounded_rect(test_rect, rg) for rg in regions)
                if hit1: break
                safe += 1
                continue
            break
        safe += d
    return safe


def _can_merge(A, B, eps=5.0):
    """相切边等长（允许 eps 容差）+ 贴合才允许合并。"""
    x, y, w, h = A
    qx, qy, qw, qh = B

    vert_align = abs(x - qx) <= eps and abs(w - qw) <= eps
    vert_touch = abs((y + h) - qy) <= eps or abs((qy + qh) - y) <= eps
    if vert_align and vert_touch:
        return True

    horz_align = abs(y - qy) <= eps and abs(h - qh) <= eps
    horz_touch = abs((x + w) - qx) <= eps or abs((qx + qw) - x) <= eps
    if horz_align and horz_touch:
        return True
    return False

def _plan_gap_tiles(covered, div, out_w, out_h, gap_max_px, overlap_px=0,
                    regions=None, target_tiles=GAP_TARGET_TILES):
    """与前端 planGaps 逐位镜像：均分 → 单刀切割 → 大块吞最小碎块 → 变胖 → 超尺寸平分。"""
    if regions is None:
        regions = []

    if not regions:
        return [(int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                for r in _balanced_subsplit([(0, 0, out_w, out_h)], gap_max_px)]

    base_grid = _balanced_subsplit([(0, 0, out_w, out_h)], gap_max_px)

    # 1) 均分 + 手绘框单刀切割（subtractSingleCut 镜像）
    pieces = []
    for b in base_grid:
        parts = [b]
        for rg in regions:
            rr_dict = rg if isinstance(rg, dict) else {
                "x": rg[0], "y": rg[1], "w": rg[2], "h": rg[3],
                "rtl": 0, "rtr": 0, "rbl": 0, "rbr": 0
            }
            nxt = []
            for p in parts:
                res = _subtract_single_cut(p, rr_dict)
                if res is not None:
                    nxt.append(res)
            parts = nxt
            if not parts:
                break
        pieces.extend(parts)

    # 2) 合并循环：每次只吞一个"面积最小的可合并邻居"，吞完立即重排、重扫
    changed = True
    iter_count = 0
    while changed and iter_count < 60:
        changed = False
        iter_count += 1
        pieces.sort(key=lambda p: -(p[2] * p[3]))  # 大块优先

        for i in range(len(pieces)):
            if pieces[i] is None:
                continue
            x, y, w, h = pieces[i]

            # ---- 2a. 收集可合并邻居，挑面积最小的吞 ----
            best_j = -1
            best_area = float("inf")
            for j in range(len(pieces)):
                if i == j or pieces[j] is None:
                    continue
                if not _can_merge((x, y, w, h), pieces[j]):
                    continue
                a = pieces[j][2] * pieces[j][3]
                if a < best_area:
                    best_area = a
                    best_j = j

            if best_j >= 0:
                qx, qy, qw, qh = pieces[best_j]
                nx = min(x, qx); ny = min(y, qy)
                nx_end = max(x + w, qx + qw)
                ny_end = max(y + h, qy + qh)
                pieces[i] = (nx, ny, nx_end - nx, ny_end - ny)
                pieces[best_j] = None
                changed = True
                break  # 重新排序、重新扫描，避免累积尺寸问题

            # ---- 2b. 无邻居可吞，且自身细长时向 A 外变胖 ----
            if max(w, h) / max(1, min(w, h)) > 5.0:
                target_w, target_h = w, h
                if w > h:
                    target_h = math.ceil(w / 5.0)
                else:
                    target_w = math.ceil(h / 5.0)

                bx, by, bw, bh = x, y, w, h

                if target_h > h:
                    diff = target_h - h
                    safe_t = _get_safe_expand((x, y, w, h), 'top',
                                              min(diff, y), regions)
                    safe_b = _get_safe_expand((x, y - safe_t, w, h + safe_t), 'bottom',
                                              min(diff - safe_t, out_h - (y + h)), regions)
                    by = y - safe_t
                    bh = h + safe_t + safe_b

                if target_w > w:
                    diff = target_w - w
                    # 用变胖后的 [bx,by,bw,bh] 再做右向探测
                    safe_l = _get_safe_expand((bx, by, bw, bh), 'left',
                                              min(diff, bx), regions)
                    safe_r = _get_safe_expand((bx - safe_l, by, bw + safe_l, bh), 'right',
                                              min(diff - safe_l, out_w - (bx + bw)), regions)
                    bx = bx - safe_l
                    bw = bw + safe_l + safe_r

                overlaps_neighbor = False
                for j in range(len(pieces)):
                    if i == j or pieces[j] is None:
                        continue
                    qx, qy, qw, qh = pieces[j]
                    ix = max(bx, qx); iy = max(by, qy)
                    ix2 = min(bx + bw, qx + qw); iy2 = min(by + bh, qy + qh)
                    if ix < ix2 - 0.1 and iy < iy2 - 0.1:
                        overlaps_neighbor = True
                        break

                if (not overlaps_neighbor) and (bx != x or by != y or bw != w or bh != h):
                    pieces[i] = (bx, by, bw, bh)
                    changed = True
                    break

        if changed:
            pieces = [p for p in pieces if p is not None]

    # 3) 超 gap_max 的块做均衡切分，同时把 float 坐标收敛成 int 以防 _balanced_subsplit 报错
    final_pieces = []
    for p in pieces:
        p_int = (int(round(p[0])), int(round(p[1])),
                 int(round(p[2])), int(round(p[3])))
        if p_int[2] > gap_max_px or p_int[3] > gap_max_px:
            final_pieces.extend(_balanced_subsplit([p_int], gap_max_px))
        else:
            final_pieces.append(p_int)
    return final_pieces



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
        src_meta = {"w": int(w), "h": int(h)}   
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
            bg = [{"filename": name, "subfolder": "", "type": "temp", **src_meta}]   
        except Exception as e:
            logging.warning("[TileSplit] preview temp failed: %s", e)
        try:
            sub = "element_tile_split/preview"
            out_dir = os.path.join(folder_paths.get_input_directory(), sub)
            os.makedirs(out_dir, exist_ok=True)
            name = f"ets_preview_{unique_id}.jpg"
            pil.save(os.path.join(out_dir, name), format="JPEG", quality=92)
            ref = {"filename": name, "subfolder": sub, "type": "input", **src_meta}  
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
                    "radius_tl": t.get("radius_tl", 0),
                    "radius_tr": t.get("radius_tr", 0),
                    "radius_bl": t.get("radius_bl", 0),
                    "radius_br": t.get("radius_br", 0),
                    "layer_region": t.get("layer_region"),
                })

        manual_mode = mode == "manual"
        covered = None

        if manual_mode:
            layer_regions = []
            scale_r = u_up if CORNER_RADIUS_SCALES_WITH_UPSCALE else 1.0
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

                half_lim = max(1, min(rw, rh) // 2)
                rtl = min(int(round(l.get("r_tl", 0) * scale_r)), half_lim)
                rtr = min(int(round(l.get("r_tr", 0) * scale_r)), half_lim)
                rbl = min(int(round(l.get("r_bl", 0) * scale_r)), half_lim)
                rbr = min(int(round(l.get("r_br", 0) * scale_r)), half_lim)

                layer_regions.append((l, {
                    "x": rx, "y": ry, "w": rw, "h": rh,
                    "rtl": rtl, "rtr": rtr, "rbl": rbl, "rbr": rbr,
                }))

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
                    
                    s["radius_tl"] = rg["rtl"]
                    s["radius_tr"] = rg["rtr"]
                    s["radius_bl"] = rg["rbl"]
                    s["radius_br"] = rg["rbr"]
                    
                    s["layer_w"] = rg["w"]
                    s["layer_h"] = rg["h"]
                    s["offset_x"] = s["x"] - rg["x"]
                    s["offset_y"] = s["y"] - rg["y"]
                    
                    s["layer_region"] = (rg["x"], rg["y"], rg["w"], rg["h"])
                emit(subs, layer=l, layer_index=li)

            if not layer_regions and not fill_gaps:
                logging.warning("[TileSplit] manual mode with no boxes; "
                                "whole canvas as one tile")
                emit([{"x": 0, "y": 0, "w": out_w, "h": out_h,
                       "col": 0, "row": 0, "kind": "fallback"}])

            if ov_mode == "px":
                default_ov = max(0, int(ov_px or 0))
            else:
                default_ov = int(round(min(out_w, out_h) * (_clamp(float(ov_pct), 0.0, 45.0) / 100.0)))

            if fill_gaps and covered is not None:
                rg_list = [rg for _, rg in layer_regions]
                # 传入 overlap_px=0，最后会对所有 tiles 统一外扩
                for i, (gx, gy, gw, gh) in enumerate(
                        _plan_gap_tiles(covered, div, out_w, out_h, gap_max,
                                        overlap_px=0, regions=rg_list)):
                    emit([{"x": gx, "y": gy, "w": gw, "h": gh,
                           "col": i, "row": 0, "kind": "gap"}])
        else:
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
            _inject_manual_overlap(tiles, default_ov, out_w, out_h)  # 加上 out_w 和 out_h
            _compute_manual_fades(tiles, default_ov, out_w, out_h)
            grid_overlap = max(
                (max(t.get("fade_top", 0), t.get("fade_bottom", 0),
                     t.get("fade_left", 0), t.get("fade_right", 0))
                 for t in tiles),
                default=0,
            )
            grid_overlap = max(grid_overlap, max(ov_values) if ov_values else 0)
        else:
            grid_overlap = max(ov_values) if ov_values else 0

        pbar = comfy.utils.ProgressBar(len(tiles))
        tile_images, grid_tiles = [], []
        for idx, t in enumerate(tiles):
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
            
            # 【新增】生成当前分块（含重叠区）基于绝对画布坐标的 Mask
            tile_mask = None
            if t.get("kind") == "layer":
                rtl = int(t.get("radius_tl", 0))
                rtr = int(t.get("radius_tr", 0))
                rbl = int(t.get("radius_bl", 0))
                rbr = int(t.get("radius_br", 0))
                if rtl or rtr or rbl or rbr:
                    lx, ly, lw, lh = t["layer_region"]
                    bx2, by2 = lx + lw, ly + lh
            
                    # 膨胀量 = _inject_manual_overlap 里用的 half
                    half = max(0, default_ov // 2)
            
                    ys = torch.arange(th, dtype=torch.float32) + 0.5 + ty
                    xs = torch.arange(tw, dtype=torch.float32) + 0.5 + tx
                    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            
                    # 膨胀框（直边外扩 half）
                    dx0, dy0 = lx - half, ly - half
                    dx1, dy1 = bx2 + half, by2 + half
                    mask_b = (xx >= dx0) & (xx <= dx1) & (yy >= dy0) & (yy <= dy1)
            
                    # 膨胀圆角：圆心不变，半径 r + half；超出即挖掉
                    if rtl > 0:
                        cx, cy = lx + rtl, ly + rtl
                        m = (xx < cx) & (yy < cy)
                        mask_b &= ~(m & ((xx - cx)**2 + (yy - cy)**2 > (rtl + half)**2))
                    if rtr > 0:
                        cx, cy = bx2 - rtr, ly + rtr
                        m = (xx > cx) & (yy < cy)
                        mask_b &= ~(m & ((xx - cx)**2 + (yy - cy)**2 > (rtr + half)**2))
                    if rbl > 0:
                        cx, cy = lx + rbl, by2 - rbl
                        m = (xx < cx) & (yy > cy)
                        mask_b &= ~(m & ((xx - cx)**2 + (yy - cy)**2 > (rbl + half)**2))
                    if rbr > 0:
                        cx, cy = bx2 - rbr, by2 - rbr
                        m = (xx > cx) & (yy > cy)
                        mask_b &= ~(m & ((xx - cx)**2 + (yy - cy)**2 > (rbr + half)**2))
            
                    tile_mask = mask_b.float().unsqueeze(0).unsqueeze(-1)

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
                "radius_tl": t.get("radius_tl", 0),
                "radius_tr": t.get("radius_tr", 0),
                "radius_bl": t.get("radius_bl", 0),
                "radius_br": t.get("radius_br", 0),
                "layer_region": t.get("layer_region"),
                "mask": tile_mask, # 将算好的像素级 mask 传入 grid
                "touch_left": tx == 0,
                "touch_top": ty == 0,
                "touch_right": tx + tw >= out_w,
                "touch_bottom": ty + th >= out_h,
            })
            pbar.update_absolute(idx + 1, len(tiles))

        grid = {
            "tiles": grid_tiles,
            "num_x": 0, "num_y": 0,          
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

