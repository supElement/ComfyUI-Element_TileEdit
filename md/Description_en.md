# Element_TileEdit Parameter Manual

---

## TiledUpscale (All-in-One Node)

Does everything in one pass: upscale → split into tiles → re-render each tile → feathered merge.

### Input Ports

| Port | Type | Required | Description |
|---|---|---|---|
| `model` | MODEL | Yes | Sampling model. Can be preceded by patch nodes. |
| `vae` | VAE | Yes | For encoding/decoding. |
| `image` | IMAGE | Yes | Source image. |
| `clip` | CLIP | Yes | Both positive/negative conditions are encoded from here. Will error out if not connected. |

### Output Ports

| Port | Type | Description |
|---|---|---|
| `image` | IMAGE | Final merged image. |
| `split_images` | IMAGE list | Per-tile source crops (before resampling). For debugging or comparison. |
| `resampler_images` | IMAGE list | Per-tile re-rendered results. |
| `grid` | GRID | Structured JSON. Already consumed internally; generally not needed. |

### Parameters

| Parameter | Type | Default | Description | Notes |
|---|---|---|---|---|
| `upscale_by` | FLOAT 1.0–16.0 | 1.0 | Upscale factor before splitting. | `1.0` skips upscaling but still splits and re-renders. >1 uses lanczos upscaling first. |
| `tile_mode` | BOOLEAN | True | ON = split by rows/columns (uses `columns/rows`); OFF = split by size (uses `tile_size`). | Toggling **hides/shows** the corresponding widgets, but hidden values are preserved (via `converted-widget`). |
| `columns` | INT 1–16 | 2 | Column count. | Only effective when `tile_mode = ON`. |
| `rows` | INT 1–16 | 2 | Row count. | Only effective when `tile_mode = ON`. |
| `tile_size` | INT 0–4096 (step 16) | 0 | Per-tile generation resolution. | Only effective when `tile_mode = OFF`. `0` = auto (about 4 tiles along the long edge, falling between 1024–1536). **For identity-edit LoRAs, keep each tile ≤ 2MP**. |
| `overlap` | INT 0–2048 (step 16) | 0 | Overlap pixels between adjacent tiles. | `0` = auto (15% of tile size, capped at 256px). **Non-zero values are used as-is, no cap**. Increase if seams are obvious. |
| `feather` | FLOAT 0–1 | 1.0 | Merge feathering strength. | Lower values → harder seams. Uses smoothstep curve, not linear. |
| `seed` | INT | 0 | Sampling seed. | Per-tile seeds derive deterministically from this; changing it affects all tiles. |
| `steps` | INT 1–10000 | 8 | Per-tile sampling steps. | The `reference_strength` list **corresponds strictly one-to-one** with this. |
| `cfg` | FLOAT | 1.0 | CFG. | **Negative conditioning is only encoded in in-context / grounded branches when > 1**. With Krea 2 in-context, usually keep at 1.0. |
| `sampler_name` | SAMPLERS | euler | Sampler. | — |
| `scheduler` | SCHEDULERS | simple | Scheduler. | — |
| `denoise` | FLOAT 0–1 | 1.0 | Re-render strength. | Transparent to internal scheduling: denoise only changes the sigma range, not the forward pass count, so it doesn't affect `reference_strength`'s per-step correspondence. |
| `reference_mode` | BOOLEAN | False | ON = Krea 2 identity-edit (`[text | source | target]` in-context layout); OFF = generic conditioning path. | **Only works with Krea 2**. On non-Krea 2 models, turning it ON **auto-falls back** to conditioning with a warning. `clip` must be connected when ON. |
| `redo_first_tile` | BOOLEAN | True | After all tiles finish, re-renders the first tile once so it can see finished neighbors. | Only effective when tile count > 1. Costs one extra sampling pass. |
| `grounding_px` | INT 0–4096 (step 32) | 768 | Text encoder vision-slot size. | Qwen Edit 2509/2511: area bucket, `0` or `≥384` = official 384×384; Qwen Image 2.1: **ignored**; Krea 2: caps the vision input's longest side, `0` = use tile size. |
| `reference_strength` | STRING | `"1.0,1.0,0.95,0.9,0.8"` | Per-step reference strength list. | Fewer values → last value repeats; more → only leading values used. Separators: `, ， 、 : ： ; ； space`. `1.0` = original behavior; `0.0` = that step has no reference; `0/1` uses binary single-forward; between `0~1` uses guidance **dual-forward** (≈cfg=2 cost). All-1 list = zero overhead. |
| `prompt` | STRING multiline | `"enhance quality preserving all the details"` | Positive condition. | **Cannot be empty** (positive port removed; this is the only source). |
| `negative_prompt` | STRING multiline | `""` | Negative condition. | Empty = encode empty string. |

### Branch Behavior (automatic, no manual selection)

| Model condition | Route | Notes |
|---|---|---|
| `reference_mode=True` and Krea 2 | in_context | `[text | source | target]` layout. |
| prompt non-empty AND tokenizer handles images AND **not** FLUX.2 family | grounded_conditioning | Per-tile raw pixels go into vision slots; no `reference_latents` attached. |
| otherwise | generic reference-latents | Each tile attaches its own latent. |

> **FLUX.2 / klein always uses the reference-latent path**; forcing grounded would break the conditioning distribution.

---

## ElementTileSplit (Split Version, Step 1)

Pure geometric splitting + image source intake, no model dependency. At execution, the backend **fully recomputes geometry**; the front-end UI is preview only.

### Input Ports

| Port | Type | Required | Description |
|---|---|---|---|
| `image` | IMAGE | No | **Takes priority over UI-uploaded image**. When unconnected, uses the UI-dragged/pasted image. |
| `layout_data` | STRING | — | Internal hidden widget holding UI state JSON. **Do not edit manually**. |

### Output Ports

| Port | Type | Description |
|---|---|---|
| `split_images` | IMAGE list | Per-tile images **already resampled to `gen_w × gen_h`** (not the raw crop). Connect to Encode. |
| `original_image` | IMAGE | Source image. Connect to Merge's `original_image`. |
| `prompt` | STRING list | Per-tile positive condition (after layer + global composition). |
| `negative_prompt` | STRING list | Per-tile negative condition. |
| `grid` | GRID | Structured JSON. Connect to Encode / Merge. |

### Three Modes

| Mode | Behavior | Notes |
|---|---|---|
| `by Manual` | Hand-drawn boxes. Each layer has its own upscale / cols / rows / prompt. | Supports `fill gaps` to auto-tile uncovered areas. |
| `by count` | Split evenly by `unified.columns × unified.rows`. | **Ignores all hand-drawn boxes**. |
| `by size` | Split by `unified.tile_size`. | **Ignores all hand-drawn boxes**. |

### UI Top Toolbar

| Control | Default | Description | Notes |
|---|---|---|---|
| `▶` | — | Run up to this node only (isolated prompt). | Doesn't run downstream, only for preview. |
| `📂` | — | Upload a local image. | — |
| `by Manual / by count / by size` | Manual | Switch mode. | Switching doesn't lose hand-drawn box data, but it's **inactive** when in count/size mode. |
| `Div` | 16 | Alignment base (2–256, step 2). | Usually leave alone. All coordinates/sizes snap to this. |
| `U-upscale` | 2.0 | Unified upscale factor (0.1–16). | Output canvas = source × this, ceil-aligned to `div`. |
| `Cols / Rows` | 2 / 2 | Count mode rows/columns. | Only active in count mode. |
| `tile` | 1024 | Target size in size mode. | Only active in size mode. |
| `overlap px / %` | 64px | Overlap. Click `px` / `%` buttons to switch units. | Both values are kept; only the currently active unit takes effect. |
| `fill gaps` | on | Auto-fill uncovered areas outside boxes. | Only effective in manual mode. |
| `max side` | 1536 | Max side length for gap-fill tiles. | Only affects fill-gap tiles. |

### UI Right Layers Panel

| Control | Description | Notes |
|---|---|---|
| `alone` | Allow per-layer independent upscale. | Unchecked = all layers use `unified.upscale`. Only when checked can per-row `upscale` be edited. |
| Per-row `upscale` | Per-layer upscale factor. | Only works when `alone` is checked. |
| Per-row `cols / rows` | Subdivide this layer into this many columns/rows. | 1×1 = no subdivision. Each resulting tile has its own fade. |
| `✕` | Delete this layer. | — |
| Drag row | Reorder layers. | Affects final tile order, not geometry. |
| `G` row | Displays gap tile count and max side. | Read-only. |
| `U` row | Read-only summary for count / size modes. | Shows tile count, not positions. |

### UI Bottom Prompt Panel

| Control | Description |
|---|---|
| `layer prompt` | Edit the **currently selected box**'s prompt. |
| `global prompt` | Edit the prompt shared by **all tiles**. |
| `add` | Checked = global appended after layer (`layer, global`); unchecked = layer if layer exists, else global. |
| `positive prompt` | Positive condition for the current editing target. |
| `negative prompt` | Negative condition for the current editing target. |

### UI Canvas Interactions

| Action | Effect |
|---|---|
| `Ctrl + drag` | Create a new layer box (snapped to div). |
| Drag inside a box | Move. |
| Drag a box's 8 handles | Resize (8px hit tolerance). |
| Drag the control points at the four corners of the box | Adjust the size of the rounded corners; each corner can be controlled independently. This does not affect the list of output tiled images, as the rounding is applied only within the Element Tile Merge node |
| Click empty area | Deselect. |
| `Delete / Backspace` | Delete currently selected layer. |
| Drag image / Ctrl+V | Upload and replace source image. |

### UI State Persistence

- State is written to both the `layout_data` widget and `localStorage` (with **workflow fingerprint**).
- Reopening the same workflow auto-restores. Different workflows don't cross-contaminate.
- At execution, the backend **fully recomputes**; stale front-end state won't pollute results.

### Prompt Composition Rules

- `global_append = True` (`add` checked): `layer, global`
- `global_append = False` (`add` unchecked): `layer if layer.strip() else global`
- **No-box layers (gap / fallback)** only output global.

### Output Notes

- `split_images` has already been edge-extended to `gen_w × gen_h`.
- Total tile count limit `MAX_TILES = 256`; exceeding it raises an error.
- If the backend finds no tiles, it degrades to "single whole-canvas tile" and logs a warning.

---

## ElementTileEncode (Split Version, Step 2)

Encodes each tile's image / prompt into per-tile LATENT + CONDITIONING.

### Input Ports

| Port | Type | Required | Description |
|---|---|---|---|
| `model` | MODEL | Yes | Used for alignment, Krea 2 detection, grounded detection. |
| `vae` | VAE | Yes | Encodes each tile image. |
| `clip` | CLIP | Yes | Encodes positive/negative conditions. |
| `tile_images` | IMAGE list | Yes | From Split's `split_images`. |
| `tile_prompts` | STRING list | Yes | From Split's `prompt`. |
| `tile_negatives` | STRING list | Yes | From Split's `negative_prompt`. |

### Output Ports

| Port | Type | Description |
|---|---|---|
| `tile_positives` | CONDITIONING list | One per tile. Connect to KSampler's positive. |
| `tile_negatives` | CONDITIONING list | One per tile. Connect to KSampler's negative. |
| `tile_latents` | LATENT list | One per tile. Connect to KSampler's latent_image. |

### Parameters

| Parameter | Type | Default | Description | Notes |
|---|---|---|---|---|
| `reference_mode` | BOOLEAN | False | ON = Krea 2 identity-edit (in-context layout); OFF = conditioning (auto-grounded depending on tokenizer capability). | **Only works with Krea 2**. When ON, non-Krea 2 falls back to conditioning with a warning. When ON, you **must** connect `Krea2 In-Context Patch` before KSampler. |
| `grounding_px` | INT 0–4096 (step 32) | 768 | Vision-slot size. | Same semantics as TiledUpscale. |

### Branch Behavior (automatic)

| Condition | Branch |
|---|---|
| `reference_mode=True` and Krea 2 | in_context (attaches `reference_latents + index_timestep_zero`) |
| `reference_mode=False` AND prompt non-empty AND tokenizer handles images AND **not** FLUX.2 | grounded |
| otherwise | generic reference-latents |

### Required Patches

**This node does not output MODEL**. To make reference strength take effect step-by-step:

| Which branch | Required patch |
|---|---|
| generic reference-latents | `ElementStrengthSchedule` (connect to KSampler's model) |
| Krea 2 in-context | `ElementKrea2InContextPatch` (optionally stack `StrengthSchedule` on top) |

---

## ElementTileMerge (Split Version, Final Step)

Stitches each tile's generated result back onto the canvas using `grid`.

### Input Ports

| Port | Type | Required | Description |
|---|---|---|---|
| `tile_images` | IMAGE list | Yes | Per-tile generated results (after KSampler → VAE Decode). |
| `grid` | GRID | Yes | From Split. |
| `feather` | FLOAT 0–1 | Yes | Feathering strength. Default 1.0. |
| `tile_sources` | IMAGE list | No | Per-tile original images. Used as color-matching reference. |
| `original_image` | IMAGE | No (**recommended**) | Underlay image. Without it, edge alpha < 1 may cause black fringing. |
| `color_match_strength` | FLOAT 0–1 | No | Color matching strength. `0` = off. |

### Output Ports

| Port | Type | Description |
|---|---|---|
| `IMAGE` | IMAGE | Merged result. |

### Parameters

| Parameter | Type | Default | Description | Notes |
|---|---|---|---|---|
| `feather` | FLOAT 0–1 | 1.0 | Feathering strength. | Uses smoothstep curve, not linear. `0` ≈ no feathering, seams will be hard. |
| `color_match_strength` | FLOAT 0–1 | 0.0 | Color matching strength. | **Only triggers when > 0**, and only truly works when **`tile_sources` is also connected**. Statistics use **low-weight edge pixels** (< 0.8) to avoid washing out center content. |

### Merge Algorithm

1. Underlay: `original_image` scaled to `work_w × work_h`.
2. Foreground (`kind = layer`) accumulates into fg canvas; background (`gap / fallback / unified`) accumulates into bg canvas.
3. Normalize then composite: `base = bg * bg_alpha + original * (1 - bg_alpha)` → `merged = fg * fg_alpha + base * (1 - fg_alpha)`.
4. Crop to `img_h × img_w` (oversampled regions are trimmed).

### Notes

- **Without `tile_sources`, `color_match_strength` is completely ineffective**.
- **When Split is in manual mode and `fill gaps` is not active**, not connecting `original_image` causes uncovered missing areas to show black underlay.
- `feather = 0` equals hard stitching.
- You can use the [Smart merge images node](https://github.com/supElement/ComfyUI_Element_easy) from another extension as a replacement; it provides better results for color and distortion correction.
---

## ElementKrea2InContextPatch (Model Patch)

**Only effective on Krea 2**. Non-Krea 2 models pass-through (returned as-is) with an info log.

### Input / Output

| Direction | Port | Description |
|---|---|---|
| in | `MODEL` | Sampling model. |
| out | `MODEL` | Patched. Connect to KSampler's model. |

### What It Does

1. `extra_conds` override: converts `reference_latents` in conditioning into the `ref_latents` forward kwarg. Without this, core drops the key and references never reach the DiT.
2. `DIFFUSION_MODEL` wrapper: rebuilds the sequence with the `[text | source(frame=1) | target(frame=0)]` in-context layout.

### When to Use

| Scenario | Needed? |
|---|---|
| All-in-one + Krea 2 + `reference_mode = ON` | **Not needed** (handled internally) |
| Split version + Krea 2 + Encode's `reference_mode = ON` | **Required** |
| Non-Krea 2 models | Harmless to connect (pass-through) |

---

## ElementStrengthSchedule (Model Patch)

Attaches a per-step `reference_strength` schedule to the model.

### Input / Output

| Direction | Port | Description |
|---|---|---|
| in | `model` | Sampling model. |
| in | `reference_strength` | STRING, per-step strength list. |
| out | `MODEL` | Patched. Connect to **KSampler's model input**. |

### Parameters

| Parameter | Type | Default | Description | Notes |
|---|---|---|---|---|
| `reference_strength` | STRING | `"1.0,1.0,1.0,0.8,0.5"` | Per-step reference strength list. | Same semantics as TiledUpscale. See details below. |

### `reference_strength` Details

- **Strictly one-to-one with `steps`**: step `i` (0-based) uses value `i`.
- Fewer values → last value repeats; more → only leading values used.
- Separators: English/Chinese comma `,，`, ideographic comma `、`, colon `:：`, semicolon `;；`, space — all work.
- Single value = all same.

### Three Working Modes (auto-detected)

| List content | Mode | Compute |
|---|---|---|
| All `>= 0.999` | Passthrough (no patch) | Zero overhead |
| All `0` or `1` | binary: `s=0` step strips reference directly | Single-pass |
| Contains decimals | guidance: `pred = pred_no_ref + s * (pred_with_ref - pred_no_ref)` | **Double** (≈cfg=2) |

### Notes

- **Must be connected to KSampler's model input**. Split doesn't output MODEL; connecting it after Split does nothing.
- **Don't use it together with TiledUpscale**: the all-in-one already attaches the same-named patch internally; re-attaching gets stripped by `_strip_owned_patches` and re-added, which is redundant.
- For decimal separators, **use the English period `0.8`**; Chinese periods will fail to parse.
- Can accept the `strength_string` output from [Element_listGraph_Curve](https://github.com/supElement/ComfyUI_Element_easy).
- Lower parameter values ​​increase the likelihood of "drift." To ensure consistency, use higher values ​​for the initial steps—ideally, set the value to 1 for the first few steps. For subsequent steps, the value can be adjusted downward depending on the specific model being used; this approach is more effective for correcting imperfections. This parameter is independent of the `denoise` setting and does not affect vision- or grounding-based references. It is not applicable to models not designed for image editing (e.g., SDXL, Qwen-Image-2512, etc.).
