# nodes_patch.py
"""Model patch that makes a Krea 2 model consume reference latents attached to
the conditioning (by Element Tile Split in `reference_mode`) using the
[text | source(frame=1) | target(frame=0)] in-context layout trained by
krea2edit-trainer (identity_edit LoRAs).

Two patches are applied:
  1. `extra_conds` override — the sampler calls model.extra_conds(**cond_dict)
     and DROPS unknown keys, so `reference_latents` never reaches the DiT
     forward on its own. We convert it to the `ref_latents` forward kwarg here
     (same trick as Ostris's Krea2OstrisEditModelPatch).
  2. a DIFFUSION_MODEL wrapper that reads that `ref_latents` kwarg and rebuilds
     the sequence in the identity-edit layout instead of ComfyUI's native
     T2I layout.
"""
import logging

import torch

from .common import (
    _krea2_in_context_forward, _is_krea2, _parse_strength_schedule, _scheduled_ref_model,
    _strip_owned_patches, _describe_ref_channels, _ELEMENT_FAMILIES, _CHANNEL_OVERRIDES,
    _AUTO_LABEL, _CHANNEL_AUTO_LABEL, _MODEL_FAMILY_KEY,
)

# 模型家族与参考通道的唯一真相在 common._ELEMENT_FAMILIES / _CHANNEL_OVERRIDES：
# 本节点据此生成下拉、取参考通道，并把家族 key 写进返回的 MODEL；下游
# ElementTileEncode 读同一个 key 决定 conditioning 分支，不再有第二个下拉。
_FAMILY_OPTIONS = [_AUTO_LABEL] + [spec["label"] for spec in _ELEMENT_FAMILIES.values()]
_FAMILY_BY_LABEL = {spec["label"]: key for key, spec in _ELEMENT_FAMILIES.items()}
_CHANNEL_OPTIONS = [_CHANNEL_AUTO_LABEL] + [spec["label"] for spec in _CHANNEL_OVERRIDES.values()]
_CHANNEL_BY_LABEL = {spec["label"]: key for key, spec in _CHANNEL_OVERRIDES.items()}

class ElementStrengthSchedule:
    """MODEL -> MODEL：把逐步 reference_strength 调度挂到模型上。
    供 ElementTileSplit + 原生 KSampler 工作流使用（Split 不输出 MODEL，
    调度必须经模型 patch 进入采样）。KSampler 的 model 输入必须接本节点输出。
    model_type 是模型家族的唯一开关：既决定本节点的参考通道，也写进返回的
    MODEL，供下游 ElementTileEncode 决定 conditioning 分支（消除第二个下拉）。"""
    CATEGORY = "Element TileEdit"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "reference_strength": ("STRING", {"default": "1.0,1.0,1.0,0.8,0.5", "multiline": False, "tooltip":
                "Per-step strengths (,，、:：; separated); short list repeats last, long list truncated; 1.0 = original; denoise ignored.\n"
                "逐 step 强度（,，、:：; 分隔）；不足重复末值，超出截断；1.0 = 原行为；与 denoise 无关。"}),
        }, "optional": {
            "model_type": (_FAMILY_OPTIONS, {"default": _AUTO_LABEL, "tooltip":
                "auto = detect from the loaded model; a manual pick also drives the downstream ElementTileEncode branch.\n"
                "auto = 自动检测；手动指定同时决定本节点通道与下游 ElementTileEncode 分支。"}),
            "channel_override": (_CHANNEL_OPTIONS, {"default": _CHANNEL_AUTO_LABEL, "tooltip":
                "Debug: force one raw channel over the family default; this node only, Encode unaffected.\n"
                "调试用：强制参考通道，覆盖家族默认；只影响本节点，不影响下游 Encode。"}),
        }}

    def patch(self, model, reference_strength, model_type=_AUTO_LABEL,
              channel_override=_CHANNEL_AUTO_LABEL):
        values = _parse_strength_schedule(reference_strength)
        family = _FAMILY_BY_LABEL.get(model_type, "auto")
        ref_channels = _ELEMENT_FAMILIES[family]["channels"] if family in _ELEMENT_FAMILIES else None
        channel = _CHANNEL_BY_LABEL.get(channel_override)
        if channel:
            ref_channels = _CHANNEL_OVERRIDES[channel]["channels"]
            logging.info("[TileSchedule] reference channel overridden: %s",
                         _describe_ref_channels(ref_channels))
        elif ref_channels:
            logging.info("[TileSchedule] reference channel for family %s: %s",
                         family, _describe_ref_channels(ref_channels))
        model = _strip_owned_patches(model, keys=("tiled_upscale_strength_schedule",),
                                     strip_extra_conds=False)
        # 家族 key 随 MODEL 传给下游：Encode 读它决定 conditioning 分支。
        model.model_options[_MODEL_FAMILY_KEY] = family
        return (_scheduled_ref_model(model, values, ref_channels),)


class ElementKrea2InContextPatch:
    CATEGORY = "Element TileEdit"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def patch(self, model):
        if not _is_krea2(model.get_model_object("diffusion_model")):
            logging.info("[Krea2InContextPatch] not a Krea 2 model, passing through.")
            return (model,)
        model = _strip_owned_patches(model, keys=("tiled_upscale_in_context_cond",
                                                  "tiled_upscale_in_context"))
        return (_in_context_model_from_conditioning(model),)


def _in_context_model_from_conditioning(model):
    import comfy.patcher_extension

    patched = model.clone()

    # ---- 1. extra_conds: conditioning dict -> forward kwargs ----
    # The base implementation drops unknown keys; convert reference_latents
    # ourselves. Guarded so that if a future core version already handles
    # them, we don't emit the key twice.
    orig_extra_conds = patched.model.extra_conds

    def extra_conds(**kwargs):
        out = orig_extra_conds(**kwargs)
        if "ref_latents" not in out:
            refs = kwargs.get("reference_latents", None)
            if refs:
                import comfy.conds
                out["ref_latents"] = comfy.conds.CONDList(
                    [patched.model.process_latent_in(lat) for lat in refs])
        return out

    extra_conds._element_krea2_extra_conds = True
    patched.add_object_patch("extra_conds", extra_conds)

    # ---- 2. forward wrapper: rebuild the identity-edit sequence ----
    state = {"routed": False, "warned": False}

    def wrapper(executor, x, timesteps, context, *args, **kwargs):
        source = kwargs.get("ref_latents", None)
        if source is None:
            # fallback: positional layouts differ between core versions
            for a in args:
                if isinstance(a, (list, tuple)) and a and torch.is_tensor(a[0]):
                    source = a
                    break
        if source is not None:
            if isinstance(source, (list, tuple)):
                source = source[0]
            if not state["routed"]:
                state["routed"] = True
                logging.info("[Krea2InContextPatch] in-context forward engaged, "
                             "source latent %s", tuple(source.shape))
        else:
            if not state["warned"]:
                state["warned"] = True
                logging.warning("[Krea2InContextPatch] no ref_latents reached the "
                                "forward on this call (cfg=1 uncond passes have none "
                                "by design; repeated warnings on cond passes mean "
                                "the refs are NOT arriving).")
            return executor(x, timesteps, context, *args, **kwargs)

        options = kwargs.get("transformer_options")
        if options is None:
            options = next((arg for arg in reversed(args) if isinstance(arg, dict)), {})
        return _krea2_in_context_forward(
            executor.class_obj, x, timesteps, context, source, options)

    options = patched.model_options.setdefault("transformer_options", {})
    comfy.patcher_extension.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        "tiled_upscale_in_context_cond", wrapper, options)
    return patched
