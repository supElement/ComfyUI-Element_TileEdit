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

from .common import _krea2_in_context_forward, _is_krea2, _parse_strength_schedule, _scheduled_ref_model, _strip_owned_patches

class ElementStrengthSchedule:
    """MODEL -> MODEL：把逐步 reference_strength 调度挂到模型上。
    供 ElementTileSplit + 原生 KSampler 工作流使用（Split 不输出 MODEL，
    调度必须经模型 patch 进入采样）。KSampler 的 model 输入必须接本节点输出。"""
    CATEGORY = "Element TileEdit"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "reference_strength": ("STRING", {"default": "1.0,1.0,1.0,0.8,0.5", "multiline": False, "tooltip":
                "The list of reference strengths per step corresponds solely to the 'steps' parameter; 'denoise' or 'sigma' values ​​have no effect."
                "Supports delimiters including commas (Chinese/English), enumeration commas (、), and colons. A single value applies to every step."
                "If fewer values ​​are provided than the number of steps, the final value is repeated; if more are provided," 
                "the list is truncated. A value of 1.0 represents the original behavior."}),
        }}

    def patch(self, model, reference_strength):
        values = _parse_strength_schedule(reference_strength)
        model = _strip_owned_patches(model, keys=("tiled_upscale_strength_schedule",),
                                     strip_extra_conds=False)
        return (_scheduled_ref_model(model, values),)


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
