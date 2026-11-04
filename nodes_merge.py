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

def _pick_device(fallback, need_bytes=0):
    """优先 ComfyUI 选定的加速设备(GPU), 不可用或显存不足时退回 fallback。"""
    try:
        import comfy.model_management as mm
        dev = mm.get_torch_device()
    except Exception:
        mm = None
        dev = torch.device("cuda") if torch.cuda.is_available() else None
    if dev is None or dev.type == "cpu":
        return fallback
    if need_bytes:
        try:
            free = mm.get_free_memory(dev) if mm is not None else torch.cuda.mem_get_info()[0]
        except Exception:
            free = None
        if free and free < need_bytes:
            return fallback
    return dev

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


_ANALYSIS_MAX_EDGE = 256
_COMMON_OFFSET_LIMIT = (10.0, 4.0, 4.0)
_GLOBAL_OFFSET_LIMIT = (14.0, 7.0, 7.0)
_TOTAL_OFFSET_LIMIT = (18.0, 8.0, 8.0)
_LOCAL_OFFSET_LIMIT = (8.0, 4.0, 4.0)

# ---- 改色保护(对齐 Smart Merge split_grid 分支) ----
# 双通道改色置信度阈值(CIELAB): a/b 距离与 L* 距离, 超过 hi 视为已改色
_EDIT_CHROMA_RANGE = (20.0, 60.0)
_EDIT_LUMA_RANGE = (9.8, 23.5)          # 8bit LAB 的 (25, 60) 换算到 L* 0..100
_EDIT_MEDIAN_SIZE = 15
_EDIT_BLUR_SIGMA = 6.2
# 已编辑内容内保留的校正比例: 采样已排除编辑像素, 但校正场仍会扩散进来,
# 满强度施加会把 AI 有意画的颜色(红衬衫)拉向背景的灰蓝 → 灰斑
_PROTECT_EDITED_SCALE = 0.25
# patch 边长 = 裁块短边比例, 夹在 MIN..MAX; 与 feather 解耦
_PROTECT_PATCH_FRAC = 0.12
_PROTECT_PATCH_MIN = 48
_PROTECT_PATCH_MAX = 160
_PROTECT_MIN_COVERAGE = 0.15
_PROTECT_MAD_TAU = (3.14, 6.0, 6.0)     # 8bit (8, 6, 6): L 通道换算到 L*
_PROTECT_MAX_SHIFT = (4.7, 7.0, 7.0)    # 8bit (12, 7, 7)
_PROTECT_GLOBAL_STRENGTH = 0.35
_PROTECT_GLOBAL_LIMIT = (3.92, 4.0, 4.0)
# 校正场是低频的: patch/边界衰减都按短边比例缩放, 在小图上求解再上采样几乎等效
_PROTECT_ANALYSIS_MAX_EDGE = 384


def _as_batch_hwc(x):
    return x.unsqueeze(0) if x.ndim == 3 else x[:1]


def _resize_hwc(x, max_edge):
    x = _as_batch_hwc(x)
    h, w = x.shape[1:3]
    edge = max(h, w)
    if edge <= max_edge:
        return x
    scale = float(max_edge) / float(edge)
    size = (max(8, round(h * scale)), max(8, round(w * scale)))
    return F.interpolate(x.permute(0, 3, 1, 2), size=size, mode="area",
                         antialias=False).permute(0, 2, 3, 1)


def _fit_hwc(x, h, w, mode="bilinear"):
    x = _as_batch_hwc(x)
    if x.shape[1:3] == (h, w):
        return x
    return F.interpolate(x.permute(0, 3, 1, 2), size=(h, w), mode=mode,
                         align_corners=False if mode == "bilinear" else None).permute(0, 2, 3, 1)


def _resize_map(x, max_edge):
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    h, w = x.shape[1:3]
    edge = max(h, w)
    if edge <= max_edge:
        return x
    scale = float(max_edge) / float(edge)
    size = (max(8, round(h * scale)), max(8, round(w * scale)))
    return F.interpolate(x.permute(0, 3, 1, 2), size=size, mode="area",
                         antialias=False).permute(0, 2, 3, 1)


def _rgb_to_lab(rgb):
    rgb = rgb.float().clamp(0.0, 1.0)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055).clamp_min(1e-8) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883

    def _f(t):
        return torch.where(t > 0.008856, t.clamp_min(1e-8).pow(1.0 / 3.0),
                           7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(x), _f(y), _f(z)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=-1)


def _lab_to_rgb(lab):
    l, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def _finv(t):
        cube = t ** 3
        return torch.where(cube > 0.008856, cube, (t - 16.0 / 116.0) / 7.787)

    x = _finv(fx) * 0.95047
    y = _finv(fy)
    z = _finv(fz) * 1.08883
    lin = torch.stack((
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.9692660 * x + 1.8760108 * y + 0.0415560 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    ), dim=-1).clamp(0.0, 1.0)
    rgb = torch.where(lin <= 0.0031308, lin * 12.92,
                      1.055 * lin.clamp_min(1e-8) ** (1.0 / 2.4) - 0.055)
    return rgb.clamp(0.0, 1.0)


def _clip_lab_shift(shift, limit):
    limit = torch.tensor(limit, dtype=shift.dtype, device=shift.device)
    return shift.clamp(-limit, limit)


def _median_shift(values):
    return torch.median(values, dim=0).values


def _median_filter_hwc(x, ksize):
    """奇数核中值滤波(B,C,H,W)，按行分块控制显存。"""
    k = int(ksize) | 1
    if k <= 1:
        return x
    pad = k // 2
    b, c, h, w = x.shape
    out = torch.empty_like(x)
    block = max(1, 4_000_000 // max(1, w * k * k))
    for y in range(0, h, block):
        y2 = min(h, y + block)
        y0s, y1s = max(0, y - pad), min(h, y2 + pad)
        tile = F.pad(x[:, :, y0s:y1s, :],
                     (pad, pad, pad - (y - y0s), pad - (y1s - y2)), mode="replicate")
        win = tile.unfold(2, k, 1).unfold(3, k, 1).reshape(b, c, y2 - y, w, k * k)
        out[:, :, y:y2, :] = torch.median(win, dim=-1).values
    return out


def _fast_median_blur_hwc(x, ksize):
    """大核中值: 半分辨率执行, 等效半径不变。用于低频置信度/色偏估计。"""
    ksize = int(ksize) | 1
    if ksize <= 5:
        return _median_filter_hwc(x.permute(0, 3, 1, 2), ksize).permute(0, 2, 3, 1)
    h, w = x.shape[1:3]
    sh, sw = max(8, h // 2), max(8, w // 2)
    small = F.interpolate(x.permute(0, 3, 1, 2), size=(sh, sw), mode="area")
    k = max(3, (ksize // 2) | 1)
    k = min(k, (min(sh, sw) - 1) | 1)
    small = _median_filter_hwc(small, k)
    return F.interpolate(small, size=(h, w), mode="bilinear",
                         align_corners=False).permute(0, 2, 3, 1)


def _fast_blur_hwc(x, sigma):
    """大 sigma 高斯: 先降采样再模糊、后上采样(B,H,W,C)。"""
    sigma = float(max(sigma, 0.0))
    if sigma <= 6.0:
        return _gaussian_blur_isotropic(x.permute(0, 3, 1, 2), sigma).permute(0, 2, 3, 1)
    h, w = x.shape[1:3]
    ds = int(min(4, max(2, round(sigma / 4.0))))
    sh, sw = max(8, h // ds), max(8, w // ds)
    small = F.interpolate(x.permute(0, 3, 1, 2), size=(sh, sw), mode="area")
    small = _gaussian_blur_isotropic(small, max(0.5, sigma / ds))
    return F.interpolate(small, size=(h, w), mode="bilinear",
                         align_corners=False).permute(0, 2, 3, 1)


def _boundary_distance_hwc(mask, band):
    """mask 内像素到边界的近似距离(像素): 半分辨率逐次腐蚀, 再上采样回原尺寸。"""
    b, h, w, _ = mask.shape
    cur = mask.float()
    step = 2 if min(h, w) >= 128 else 1
    if step > 1:
        cur = F.avg_pool2d(cur.permute(0, 3, 1, 2), step).permute(0, 2, 3, 1)
        cur = (cur >= 0.5).float()
    dist = torch.zeros_like(cur)
    for _ in range(int(max(1, round(band / step)))):
        cur = -F.max_pool2d(-cur.permute(0, 3, 1, 2), 3, 1, 1).permute(0, 2, 3, 1)
        dist += (cur > 0.5).float()
    if step > 1:
        dist = F.interpolate(dist.permute(0, 3, 1, 2), size=(h, w), mode="bilinear",
                             align_corners=False).permute(0, 2, 3, 1) * step
    return dist


def _tile_mask_2d(info):
    mask = info.get("mask")
    if mask is None:
        return None
    m = mask.float()
    if m.ndim == 4:
        m = m[0, ..., 0]
    elif m.ndim == 3:
        m = m[0] if m.shape[0] == 1 else m[..., 0]
    if m.ndim != 2:
        m = m.reshape(m.shape[-2], m.shape[-1])
    return m


def _chroma_edit_confidence(generated_rgb, reference_rgb):
    """返回未改色置信度。

    刻意改色是强烈且空间连贯的色度变化；曝光/白平衡漂移则主要改变亮度。
    因此只看 a/b 的原始差值，不扣除整块偏移——这样即使改色区域占了大半块
    （此时整块中值本身就被改色带偏）也能正确识别。
    """
    lab_g = _rgb_to_lab(generated_rgb[..., :3])
    lab_r = _rgb_to_lab(reference_rgb[..., :3])
    chroma = (lab_g[..., 1:3] - lab_r[..., 1:3]).norm(dim=-1, keepdim=True)
    h, w = chroma.shape[1:3]
    sigma = max(1.0, min(h, w) / 128.0)
    chroma = _gaussian_blur_isotropic(chroma.permute(0, 3, 1, 2), sigma).permute(0, 2, 3, 1)
    return ((28.0 - chroma) / 16.0).clamp(0.0, 1.0)


def _edit_confidence(generated_rgb, reference_rgb, valid=None):
    """未改色置信度(强色度或强亮度改变都降低置信度), 高 = 未编辑。"""
    lab_g = _rgb_to_lab(generated_rgb[..., :3])
    lab_r = _rgb_to_lab(reference_rgb[..., :3])
    d_ab = (lab_g[..., 1:3] - lab_r[..., 1:3]).norm(dim=-1, keepdim=True)
    d_l = (lab_g[..., 0:1] - lab_r[..., 0:1]).abs()

    def _curve(diff, lo, hi):
        diff = _fast_median_blur_hwc(diff, _EDIT_MEDIAN_SIZE)
        diff = _fast_blur_hwc(diff, _EDIT_BLUR_SIGMA)
        return ((hi - diff) / (hi - lo)).clamp(0.0, 1.0)

    conf = torch.minimum(_curve(d_ab, *_EDIT_CHROMA_RANGE),
                         _curve(d_l, *_EDIT_LUMA_RANGE))
    if valid is not None:
        valid = valid.to(device=conf.device, dtype=conf.dtype)
        conf = conf * valid + (1.0 - valid)
    return conf


def _protect_edited_scale(conf, h, w, sigma):
    """未编辑置信度 → 校正强度系数: 未编辑处 1, 已编辑处 _PROTECT_EDITED_SCALE。"""
    em = conf if conf.shape[1:3] == (h, w) else _fit_hwc(conf, h, w)
    em = _fast_blur_hwc(em, sigma).clamp(0.0, 1.0)
    return _PROTECT_EDITED_SCALE + (1.0 - _PROTECT_EDITED_SCALE) * em


def _estimate_common_lab_shift(triples):
    """跨块估计共有 LAB 偏差，各块等权以避免大区域主导。"""
    shifts = []
    for info, generated, reference in triples:
        if reference is None:
            continue
        gen = _resize_hwc(_as_batch_hwc(generated).float()[..., :3], _ANALYSIS_MAX_EDGE)
        ref = _resize_hwc(_as_batch_hwc(reference).float()[..., :3], _ANALYSIS_MAX_EDGE)
        # 两侧设备可能不同: split 参考块跟随输入设备(GPU), VAE 解码块默认在中间设备(CPU)
        ref = ref.to(device=gen.device, dtype=gen.dtype)
        if gen.shape[1:3] != ref.shape[1:3]:
            ref = _fit_hwc(ref, gen.shape[1], gen.shape[2])
        diff = _rgb_to_lab(ref) - _rgb_to_lab(gen)
        m = _tile_mask_2d(info)
        if m is not None:
            m = m.to(device=diff.device)
            valid = _fit_hwc(m[None, ..., None], diff.shape[1], diff.shape[2],
                             mode="nearest")[0, ..., 0] > 0.5
        else:
            valid = torch.ones(diff.shape[1:3], dtype=torch.bool, device=diff.device)
        if int(valid.sum()) < 300:
            continue
        # 只采信未改色的像素，避免刻意的局部改色把共有偏移带偏
        keep = valid & (_chroma_edit_confidence(gen, ref)[0, ..., 0] > 0.5)
        if int(keep.sum()) < 300:
            # 几乎没有可信像素（多半是整块统一偏色，而非局部改色）才退回全图统计
            keep = valid
        values = diff[0][keep]
        if values.numel() >= 300:
            shifts.append(_median_shift(values))
    if not shifts:
        return None
    stacked = torch.stack(shifts)
    # 用插值中位数，避免偶数块时退化为偏低的那个值
    common = _clip_lab_shift(torch.quantile(stacked, 0.5, dim=0), _COMMON_OFFSET_LIMIT)
    if float(common.abs().max()) < 0.2:
        return None
    return common


def _estimate_protected_lab_shift(triples):
    """改色保护下的共有色偏: 只统计未编辑像素, 低强度限幅, 作为粗校正。"""
    shifts = []
    for info, generated, reference in triples:
        if reference is None:
            continue
        gen = _resize_hwc(_as_batch_hwc(generated).float()[..., :3], _ANALYSIS_MAX_EDGE)
        ref = _resize_hwc(_as_batch_hwc(reference).float()[..., :3], _ANALYSIS_MAX_EDGE)
        # 两侧设备可能不同: split 参考块跟随输入设备(GPU), VAE 解码块默认在中间设备(CPU)
        ref = ref.to(device=gen.device, dtype=gen.dtype)
        if ref.shape[1:3] != gen.shape[1:3]:
            ref = _fit_hwc(ref, gen.shape[1], gen.shape[2])
        m = _tile_mask_2d(info)
        if m is not None:
            valid = _fit_hwc(m[None, ..., None].to(device=gen.device), gen.shape[1], gen.shape[2],
                             mode="nearest")[0, ..., 0] > 0.5
        else:
            valid = torch.ones(gen.shape[1:3], dtype=torch.bool, device=gen.device)
        valid = valid & (_edit_confidence(gen, ref)[0, ..., 0] > 0.5)
        if int(valid.sum()) < 100:
            continue
        shifts.append(_median_shift((_rgb_to_lab(ref) - _rgb_to_lab(gen))[0][valid]))
    if not shifts:
        return None
    shift = _median_shift(torch.stack(shifts)) * _PROTECT_GLOBAL_STRENGTH
    shift = _clip_lab_shift(shift, _PROTECT_GLOBAL_LIMIT)
    if float(shift.abs().max()) < 0.1:
        return None
    return shift


def _luminance_color_sync(lab_g, lab_r, mask, conf):
    """仅平移 L: 未编辑区按 edited_scale 施加; 全块无未编辑像素时只收一致的亮度差。"""
    sample = mask * (conf > 0.5).float()
    fallback = int((sample > 0.5).sum()) < 100
    if fallback:
        sample = mask
    sel = sample[0, ..., 0] > 0.5
    if int(sel.sum()) < 100:
        return lab_g
    differences = (lab_r[..., 0] - lab_g[..., 0])[sel[None]]
    shift = torch.median(differences)
    if fallback:
        # 全块一致的亮度偏移是光照差，不是局部改色；非均匀差异继续保护
        if float(torch.median((differences - shift).abs())) > 1.18:
            return lab_g
        lab_g = lab_g.clone()
        lab_g[..., 0] += shift
        return lab_g
    scale = _protect_edited_scale(conf, lab_g.shape[1], lab_g.shape[2], 8.0)
    lab_g = lab_g.clone()
    lab_g[..., 0] += shift * scale[..., 0]
    return lab_g


def _boundary_color_sync(lab_g, lab_r, mask, conf, band_width, strength):
    """边界附近的平滑色差校正: 方差定可靠性, 离边界越远越弱, 编辑区衰减。"""
    h, w = lab_g.shape[1:3]
    sample = mask * (conf > 0.5).float()
    sel = sample[0, ..., 0] > 0.5
    if int((mask > 0.5).sum()) < 100 or int(sel.sum()) < 100:
        return lab_g
    med_size = max(7, (band_width // 2) | 1)
    lab_g_med = _fast_median_blur_hwc(lab_g, med_size)
    lab_r_med = _fast_median_blur_hwc(lab_r, med_size)
    masked_diff = (lab_r_med - lab_g_med) * sel[None, ..., None].float()
    gsigma = max(2.0, 0.15 * min(max(31, (band_width * 3) | 1), 151))
    diff_smooth = _fast_blur_hwc(masked_diff, gsigma)
    diff_sqr = _fast_blur_hwc(masked_diff ** 2, gsigma)
    variance = (diff_sqr - diff_smooth ** 2).clamp_min(0.0)
    pos = variance[:, ::4, ::4, :]
    pos = pos[pos > 0]
    denom = torch.median(pos) if pos.numel() > 0 else torch.tensor(1.0, device=lab_g.device)
    reliability = torch.exp(-variance / (denom + 1e-6)) * sel[None, ..., None].float()
    correction = _fast_blur_hwc(masked_diff * reliability, gsigma) \
        / (_fast_blur_hwc(reliability, gsigma) + 1e-6)
    decay = 0.4 + 0.6 * torch.exp(-_boundary_distance_hwc(mask, band_width)
                                  / max(float(band_width), 1.0))
    correction = correction * _protect_edited_scale(conf, h, w, max(4.0, band_width / 4.0))
    soft = _fast_blur_hwc(mask, max(2.0, 0.15 * (band_width | 1)))
    strength_vec = torch.tensor((0.6, 1.0, 1.0), dtype=lab_g.dtype,
                                device=lab_g.device) * strength
    return lab_g + correction * decay * soft * strength_vec


def _patch_color_sync(lab_g, lab_r, mask, conf, strength):
    """低频 patch 色差中值校正: 覆盖度 + MAD 置信, 归一化填充, 限幅; 编辑区衰减。"""
    h, w = lab_g.shape[1:3]
    sample = mask * (conf > 0.5).float()
    sel = sample[0, ..., 0] > 0.5
    if int((mask > 0.5).sum()) < 100 or int(sel.sum()) < 100:
        return lab_g
    patch = int(round(min(h, w) * _PROTECT_PATCH_FRAC))
    patch = max(_PROTECT_PATCH_MIN, min(patch, _PROTECT_PATCH_MAX, h, w))
    overlap = min(max(16, patch // 2), patch - 1)
    step = max(1, patch - overlap)
    lp_sigma = max(2.0, patch / 4.0)
    diff = _fast_blur_hwc(lab_r, lp_sigma) - _fast_blur_hwc(lab_g, lp_sigma)

    ys = list(range(0, h - patch + 1, step))
    xs = list(range(0, w - patch + 1, step))
    if not ys or ys[-1] + patch < h:
        ys.append(max(0, h - patch))
    if not xs or xs[-1] + patch < w:
        xs.append(max(0, w - patch))
    ys, xs = sorted(set(ys)), sorted(set(xs))

    # 逐块统计在 CPU 上做: 放到 GPU 会让每个中值/覆盖度判断都触发同步, 反而更慢
    dev = lab_g.device
    work = torch.device("cpu") if dev.type != "cpu" else dev
    diff_w = diff[0].to(work)
    sel = sel.to(work)
    hann = torch.hann_window(patch, periodic=False, dtype=torch.float32, device=work)
    hann = hann[:, None] * hann[None, :]
    tau = torch.tensor(_PROTECT_MAD_TAU, dtype=torch.float32, device=work)
    weight_map = torch.zeros(h, w, dtype=torch.float32, device=work)
    correction = torch.zeros(1, h, w, 3, dtype=torch.float32, device=work)
    valid_patches = 0

    for y in ys:
        for x in xs:
            y2, x2 = y + patch, x + patch
            sel_patch = sel[y2 - patch:y2, x2 - patch:x2]
            if float(sel_patch.float().mean()) < _PROTECT_MIN_COVERAGE:
                continue
            sel_stat = sel_patch[::2, ::2]
            if int(sel_stat.sum()) >= 30:
                diff_patch = diff_w[y2 - patch:y2:2, x2 - patch:x2:2]
            else:
                diff_patch = diff_w[y2 - patch:y2, x2 - patch:x2]
                sel_stat = sel_patch
            if int(sel_stat.sum()) < 30:
                continue
            values = diff_patch[sel_stat]
            med = torch.median(values, dim=0).values
            mad = torch.median((values - med).abs(), dim=0).values
            patch_conf = float(torch.exp(-(mad / tau).pow(2)).min())
            if patch_conf < 0.02:
                continue
            patch_weight = hann * sel_patch.float() * patch_conf
            correction[0, y2 - patch:y2, x2 - patch:x2] += med * patch_weight[..., None]
            weight_map[y2 - patch:y2, x2 - patch:x2] += patch_weight
            valid_patches += 1

    if valid_patches == 0:
        return lab_g
    if work != dev:
        weight_map = weight_map.to(dev)
        correction = correction.to(dev)
    valid = weight_map > 1e-8
    field = correction / weight_map[..., None].clamp_min(1e-8)
    fill_sigma = max(4.0, patch / 2.0)
    for c in range(3):
        valid_f = valid[None, ..., None].float()
        filled = _fast_blur_hwc(field[..., c:c + 1] * valid_f, fill_sigma) \
            / _fast_blur_hwc(valid_f, fill_sigma).clamp_min(1e-6)
        field[..., c] = torch.where(valid, field[..., c], filled[..., 0])
    field = _fast_blur_hwc(field, max(2.0, patch / 3.0))
    max_shift = torch.tensor(_PROTECT_MAX_SHIFT, dtype=torch.float32, device=lab_g.device)
    field = field.clamp(-max_shift, max_shift)
    scale = _protect_edited_scale(conf, h, w, max(4.0, patch / 4.0))
    return lab_g + field * scale * strength


def _protected_color_match(g, r, strength, weight, content_mask, global_shift):
    """Smart Merge split_grid 保护路径: 只采信未编辑像素的低幅校正。"""
    h, w = g.shape[1:3]
    mask = content_mask if content_mask is not None else weight
    if mask is None:
        mask = torch.ones(1, h, w, 1, dtype=g.dtype, device=g.device)
    else:
        mask = _as_batch_hwc(mask).float()
        if mask.shape[-1] != 1:
            mask = mask[..., :1]
        mask = _fit_hwc(mask, h, w).clamp(0.0, 1.0)

    # 全部在缩小后的图上求解, 最后把校正场放大回原尺寸
    g_s = _resize_hwc(g[..., :3], _PROTECT_ANALYSIS_MAX_EDGE)
    r_s = _resize_hwc(r[..., :3], _PROTECT_ANALYSIS_MAX_EDGE)
    if r_s.shape[1:3] != g_s.shape[1:3]:
        r_s = _fit_hwc(r_s, g_s.shape[1], g_s.shape[2])
    hs, ws = g_s.shape[1:3]
    mask_s = _fit_hwc(mask, hs, ws).clamp(0.0, 1.0)

    lab_g0 = _rgb_to_lab(g_s)
    lab_r = _rgb_to_lab(r_s)
    conf = _edit_confidence(g_s, r_s, valid=mask_s)
    lab_g = lab_g0

    if global_shift is not None:
        shift = global_shift.to(device=g.device, dtype=torch.float32)
        lab_g = lab_g + shift * _protect_edited_scale(conf, hs, ws, 8.0) * strength

    if int(((mask_s * (conf > 0.5).float()) > 0.5).sum()) < 100:
        lab_g = _luminance_color_sync(lab_g, lab_r, mask_s, conf)
    else:
        patch = int(round(min(hs, ws) * _PROTECT_PATCH_FRAC))
        patch = max(_PROTECT_PATCH_MIN, min(patch, _PROTECT_PATCH_MAX, hs, ws))
        lab_g = _boundary_color_sync(lab_g, lab_r, mask_s, conf, max(15, patch // 3), strength)
        lab_g = _patch_color_sync(lab_g, lab_r, mask_s, conf, strength)

    if (hs, ws) == (h, w):
        lab_out = lab_g
    else:
        lab_out = _rgb_to_lab(g[..., :3]) + _fit_hwc(lab_g - lab_g0, h, w)
    out = g.clone()
    out[..., :3] = _lab_to_rgb(lab_out)
    return out.clamp(0.0, 1.0).to(g.dtype)


def _color_match(generated, reference, strength=1.0, weight=None,
                 color_protection=False, common_shift=None, content_mask=None):
    if strength <= 0.0 or reference is None:
        return generated
    g = _as_batch_hwc(generated).float()
    r = _as_batch_hwc(reference).float()
    if r.shape[1:3] != g.shape[1:3]:
        r = F.interpolate(r.permute(0, 3, 1, 2), size=g.shape[1:3],
                          mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    if g.shape[-1] < 3 or r.shape[-1] < 3:
        return generated

    if color_protection:
        return _protected_color_match(g, r, strength, weight, content_mask, common_shift)

    h_full, w_full = g.shape[1:3]
    g_small = _resize_hwc(g[..., :3], _ANALYSIS_MAX_EDGE)
    r_small = _resize_hwc(r[..., :3], _ANALYSIS_MAX_EDGE)
    if r_small.shape[1:3] != g_small.shape[1:3]:
        r_small = _fit_hwc(r_small, g_small.shape[1], g_small.shape[2])
    h, w = g_small.shape[1:3]

    if weight is not None:
        weight = _as_batch_hwc(weight).float()
        if weight.shape[-1] != 1:
            weight = weight[..., :1]
        if weight.shape[1:3] != (h_full, w_full):
            weight = _fit_hwc(weight, h_full, w_full)
        weight_small = _resize_map(weight, _ANALYSIS_MAX_EDGE)
        if weight_small.shape[1:3] != (h, w):
            weight_small = _fit_hwc(weight_small, h, w)
    else:
        weight = None
        weight_small = torch.ones(1, h, w, 1, dtype=g.dtype, device=g.device)

    valid_soft = weight_small[..., 0].clamp(0.0, 1.0)
    trusted = valid_soft > 0.25
    if int(trusted.sum()) < 100:
        trusted = valid_soft > 0.01
    if int(trusted.sum()) < 100:
        return generated

    lab_g = _rgb_to_lab(g_small)
    lab_r = _rgb_to_lab(r_small)
    diff = (lab_r - lab_g)[trusted]
    tile_offset = _median_shift(diff)

    if common_shift is None:
        offset = tile_offset
    else:
        common = common_shift.to(device=g.device, dtype=torch.float32)
        # 本块可信像素多时以自身估计为准（每块都有对应源图，自身估计最准），
        # 只有内容稀少/被遮罩的块才更多依赖跨块共有偏移来稳定结果
        coverage = float(valid_soft.mean())
        tile_weight = min(1.0, 0.75 + 0.5 * coverage)
        offset = common + tile_weight * (tile_offset - common)
    offset = _clip_lab_shift(offset, _GLOBAL_OFFSET_LIMIT)

    l_g = lab_g[..., 0][trusted]
    l_r = lab_r[..., 0][trusted]
    g_span = torch.quantile(l_g, 0.9) - torch.quantile(l_g, 0.1)
    r_span = torch.quantile(l_r, 0.9) - torch.quantile(l_r, 0.1)
    l_scale = (r_span / g_span.clamp_min(1.0)).clamp(0.75, 1.33)

    global_delta = torch.zeros_like(lab_g)
    global_delta[..., 0] = (lab_g[..., 0] - torch.median(l_g)) * (l_scale - 1.0) + offset[0]
    global_delta[..., 1] = offset[1]
    global_delta[..., 2] = offset[2]

    matched = lab_g + global_delta
    residual = lab_r - matched
    tau = torch.tensor((8.0, 5.0, 5.0), dtype=g.dtype, device=g.device)
    reliability = valid_soft[..., None] * torch.exp(-(residual / tau).pow(2))
    sigma = max(1.0, min(h, w) / 128.0)
    num = _gaussian_blur_isotropic((residual * reliability).permute(0, 3, 1, 2), sigma)
    den = _gaussian_blur_isotropic(reliability.permute(0, 3, 1, 2), sigma)
    local_delta = (num / den.clamp_min(1e-6)).permute(0, 2, 3, 1)
    local_delta = _clip_lab_shift(local_delta, _LOCAL_OFFSET_LIMIT)

    total_delta = _fit_hwc(global_delta, h_full, w_full, mode="bilinear") \
        + 0.8 * _fit_hwc(local_delta, h_full, w_full, mode="bilinear")

    if weight is not None:
        apply_w = strength * (0.8 + 0.2 * (1.0 - weight)).clamp(0.0, 1.0)
    else:
        apply_w = torch.full((1, h_full, w_full, 1), strength,
                             dtype=g.dtype, device=g.device)

    total_delta = _clip_lab_shift(total_delta, _TOTAL_OFFSET_LIMIT)
    lab_out = _rgb_to_lab(g[..., :3]) + total_delta * apply_w
    out = g.clone()
    out[..., :3] = _lab_to_rgb(lab_out)
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


def _apply_edit_mask(weight, info, device, dtype):
    """乘上 split 的锁边 edit_mask：锁边带权重清零，合并时只取真正重绘过的内容。"""
    edit_mask = info.get("edit_mask")
    if edit_mask is None:
        return weight
    if not torch.is_tensor(edit_mask):
        return weight
    w_h, w_w = weight.shape[1], weight.shape[2]
    edit_mask = edit_mask.to(device=device, dtype=dtype)
    if edit_mask.ndim == 3:
        edit_mask = edit_mask[:1].unsqueeze(-1)
    elif edit_mask.ndim == 2:
        edit_mask = edit_mask[None, :, :, None]
    else:
        edit_mask = edit_mask[:1]
        if edit_mask.shape[-1] != 1:
            edit_mask = edit_mask[..., :1]
    if edit_mask.shape[1] != w_h or edit_mask.shape[2] != w_w:
        edit_mask = F.interpolate(edit_mask.permute(0, 3, 1, 2), size=(w_h, w_w),
                                  mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return weight * edit_mask.clamp(0.0, 1.0)


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
                "feather": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Transition width as a fraction of the overlap, centered on the overlap midline: "
                               "0 = hard switch exactly at the midline, 0.5 = only the middle half of the overlap blends, "
                               "1 = the whole overlap blends."}),
                "color_match_strength": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_protection": ("BOOLEAN", {"default": False,
                    "label_on": "true", "label_off": "false",
                    "tooltip": "On: compare each tile against its source and only damp color matching where the pixels were intentionally recolored, "
                               "so hand-painted hue/color edits survive while exposure fixes still apply. "
                               "Off: color match every tile as usual."}),
            },
            "optional": {
                "tile_sources": ("IMAGE", {"tooltip": "The unedited split tiles, same order as tile_images. Used as the color reference "
                                                       "and to tell intentional recoloring apart from exposure/white-balance drift; "
                                                       "without it color matching is skipped."}),
                "original_image": ("IMAGE", {"tooltip": "Full original image used as an underlay for canvas areas not covered by any tile."}),
                "high_low_freq": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable high-low frequency fusion: use feathered weights to smooth the transition for low frequencies, "
                               " and competitive weights to preserve details for high frequencies—preventing details in overlapping regions from being flattened out through averaging."}),
            },
        }

    def merge(self, tile_images, grid, feather, color_match_strength=0.4,
              color_protection=False,
              tile_sources=None, original_image=None, high_low_freq=False):
        grid = _unwrap(grid)
        feather = float(_unwrap(feather, 1.0))
        high_low_freq = bool(_unwrap(high_low_freq, False)) 
        cm_strength = float(_unwrap(color_match_strength, 0.4))
        color_protection = bool(_unwrap(color_protection, False))
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
        dst_device, dtype = first.device, first.dtype
        # 画布 + 权重 + 高低频缓冲约 4 倍画布大小, 留一点余量再上 GPU
        canvas_bytes = work_h * work_w * channels * 4
        device = _pick_device(dst_device, int(canvas_bytes * 4 + 512 * 1024 * 1024))
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
                w_full = _tile_weight(info_wm_copy, feather, device, dtype, centered=True)
                w_full = _apply_edit_mask(w_full, info_wm, device, dtype)
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

        if cm_strength > 0.0:
            common_shift = (_estimate_protected_lab_shift(triples) if color_protection
                            else _estimate_common_lab_shift(triples))
        else:
            common_shift = None

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
                
            weight = _tile_weight(info, feather, device, dtype, centered=True)
            weight = _apply_edit_mask(weight, info, device, dtype)

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
                decoded_c = _color_match(
                    decoded_c, src_c, cm_strength, weight_c,
                    color_protection=color_protection, common_shift=common_shift,
                    content_mask=tile_mask_c)

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
        return (merged[:, :img_h, :img_w, :].clamp(0.0, 1.0).to(device=dst_device),)

