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
| `feather` | FLOAT 0–1 | 1.0 | Merge feathering strength. | Lower values → harder seams. Uses smoothstep curve, not linear. **The all-in-one ramps from the tile edge**, unlike the split version's Merge, whose transition lives between the two tiles' locked bands. |
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

<img width="1005" height="1145" alt="image" src="https://github.com/user-attachments/assets/9785fe15-b11a-4789-a05e-f09e48b97855" />

---

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
| `tile_masks` | MASK list | Per-tile sampling mask. Not all-white when `edge mask` is on; **must** be connected to Encode's `tile_masks` (Encode declares it as a required port). |

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
| `Div` | 16 | Alignment base (2–256, step 2). | Usually leave alone. All coordinates/sizes snap to this; **the input image itself does not need to be divisible by `div`**. |
| `U-upscale` | 2.0 | Unified upscale factor (0.1–16). | Output size = `round(source × this)`, independent of `div`; the tile canvas is ceil-aligned to `div` internally only, and the generation resolution is rounded to `div`. |
| `Cols / Rows` | 2 / 2 | Count mode rows/columns. | Only active in count mode. |
| `tile` | 1024 | Target size in size mode. | Only active in size mode. |
| `overlap px / %` | 64px | Overlap. Click `px` / `%` buttons to switch units. | Both values are kept; only the currently active unit takes effect. **No upper cap** — whatever you enter is used. |
| `edge mask` | off | Lock the outer overlap band with a mask: sampling never repaints it, it's reference only; Merge drops it automatically. | Split's `tile_masks` must be wired into Encode (that port is required, so a missing wire errors out immediately). See "Edge Mask" below. |
| `lock` | 90% | Locked band width = `overlap/2 × lock%` (0–100%). | Default 90. At 100 the whole half-overlap is locked; Merge's exclusion saturates at `overlap/2`, leaving almost no repainted content in the overlap — generally not recommended as a default. |
| `feather` | 50% | Feather **inside** the locked band, as a percentage of its width (0–100%). | The outer edge stays at mask=0 and the ramp reaches 1 at the band's inner edge, so it never takes repaintable area outside the band. 100% = the ramp covers the whole band. Merge drops only the locked band itself. |
| `fill gaps` | on | Auto-fill uncovered areas outside boxes. | Only effective in manual mode. |
| `max side` | 1536 | Max side length for gap-fill tiles. | Only affects fill-gap tiles. |

### Edge Mask

When enabled, only edges that **have a neighbor** get a locked band (canvas borders are not locked):

- **Locked band** = `overlap/2 × lock%`, converted to generation resolution and written into the sampling `noise_mask`: that strip is not repainted, only used as reference. It prevents content drift under heavy restyling.
- **`feather`** is a percentage **inside** the locked band (0–100% of that edge's band width). It erodes toward mask=0: the outer edge always stays 0, the mask rises toward the tile interior and reaches 1 exactly at the band's inner edge, so **it never takes repaintable area outside the band**. `0%` = hard edge; `50%` = outer half solidly locked, inner half a gradient; `100%` = the whole band is a gradient.
- **Merge drops the locked band itself** (`≤ overlap/2`) and no longer adds `feather`, since the feathering now lives entirely inside the band. The exclusion is a hard cut; blending is handled by Merge's own `feather`.
- Each side drops at most `overlap/2 − 1px`. At `lock=100%` the band reaches the midline, is clamped, and logs a warning; almost no middle remains for Merge's feather, so it is generally not recommended.
- **Rounded hand-drawn tiles lock their corners too**: the lock band wraps around the rounded corner. The sampling silhouette is the expanded outline shrunk inward by the lock band (a **concentric** inset: same center, radius reduced from `r + overlap/2` to `r + overlap/2 − band`), so `mask=0` at the corner is a band as wide as the straight edges instead of the whole corner being cut into a large black wedge. Boxes with no rounding (radius 0) keep the original four-edge band logic. Same as the straight bands: inside the silhouette the mask stays 1 (repaintable) and the feather falls on the **outside** of the silhouette, ramping from the arc outward to 0 over the same `feather` width (no hard cut). A corner joins two edges; it is inset by the larger of their bands, and an edge with no neighbour (canvas border) contributes a zero-width band.
- Examples: `overlap=100px, lock=50%` → band 25px; `feather=100%` → those 25px are a full gradient (0 at the outer edge → 1 at the inner edge); `feather=50%` → outer 12.5px solid, inner 12.5px gradient.
- Chain: Split's `tile_masks` → Encode's `tile_masks`, **required**. Encode declares this port as required, so ComfyUI errors out at queue time instead of finishing a full sampling pass before you notice the lock was inactive.

---

The illustration below uses Tile01 as an example, with Tile02 being a mirrored version. The "feather" indication applies only to the Split node; for a Merge node, feathering uses the repaintable middle between the locked bands of two hand-drawn tiles of the same layer, keeps the 50% point on the hand-drawn box edge while stopping at this tile's exclusion boundary on a side facing the background, and transitions across a rounded corner as a radial ramp centred on the drawn arc.

<img width="2168" height="1133" alt="image" src="https://github.com/user-attachments/assets/3bf328fb-66ef-48bc-ac77-df8c6cdedacb" />

---

### Model Compatibility

Edge masking is not tied to Qwen Image 2.1: **any model sampled through ComfyUI's standard sampler works** (Qwen Image 2.1, Qwen Edit 2509/2511, Krea 2, FLUX.1, FLUX.2/klein, SD3.5, SDXL, …):

- Sampler side: Split/Encode only write the locked band into the latent's `noise_mask`; ComfyUI keeps the input latent inside that band at every step, independent of model architecture. Only third-party custom samplers that bypass `noise_mask` are unaffected.
- Merge side: dropping the locked band and the transition between the two bands are pure geometry, fully model-agnostic.
- **Resolution difference**: the quantization of the band equals the model's latent downscale factor. 8× models (Qwen Edit 2509/2511, Krea 2, FLUX.1, SD3/SDXL) give roughly 8px per cell; 16× models (Qwen Image 2.1, FLUX.2/klein) give roughly 16px per cell. At the same `lock`, the band is softer on 16× models — keep the band at least 2 latent cells wide (≥16px for 8× models, ≥32px for 16× models).
- **Boundary residual width is model-dependent**: even with feathering, a thin strip near the mask boundary may remain not fully repainted. Identity-edit / in-context models (Krea 2) and high-denoise settings leave a wider strip; inpaint-trained models follow the mask more tightly. That is why Split's `feather` and Merge's `feather` are both exposed.
- Models without inpaint/mask training (FLUX.1 dev, SD3.5, SDXL base, …) still work, but the style transition at the boundary may be harsher — increase `overlap` (and lower `lock` a little to leave a middle section for Merge's feather).

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
- **The input image does not need to be divisible by `div`**: the canvas is ceil-aligned to `div` and any excess edge is absorbed by shifting the edge tile inward as a whole (its crop comes entirely from real pixels); only a single whole-image tile that is flush with the canvas edge and cannot shift is padded with replicated edge pixels up to the canvas size, and Merge still crops back to the real size.
- **The Merge output size is always `round(source × U-upscale)`**: at `U-upscale = 1.0` it matches the original image size exactly, with no extra pixels from `div` rounding.
- **`div` only constrains each tile's generation resolution**: tiled blocks use `round_div(crop × upscale)`; a single whole-image tile is generated at the canvas size (1924 → 1936) and cropped back to 1924. The generation resolution is always a multiple of `div`, but it never changes the output size.
- Total tile count limit `MAX_TILES = 256`; exceeding it raises an error.
- If the backend finds no tiles, it degrades to "single whole-canvas tile" and logs a warning.

---

## ElementTileEncode (Split Version, Step 2)

Encodes each tile's image / prompt into per-tile LATENT + CONDITIONING, and also returns a MODEL ready for sampling (the Krea 2 in-context branch auto-applies the required patch).

### Input Ports

| Port | Type | Required | Description |
|---|---|---|---|
| `model` | MODEL | Yes | Used for alignment, Krea 2 detection, grounded detection. |
| `vae` | VAE | Yes | Encodes each tile image. |
| `clip` | CLIP | Yes | Encodes positive/negative conditions. |
| `tile_images` | IMAGE list | Yes | From Split's `split_images`. |
| `tile_prompts` | STRING list | Yes | From Split's `prompt`. |
| `tile_negatives` | STRING list | Yes | From Split's `negative_prompt`. |
| `tile_masks` | MASK list | **Yes** | From Split's 6th output, `tile_masks`. White = repaint, black = keep as reference. When `edge mask` is off it is an all-white list, so wiring it is harmless; **without it ComfyUI refuses to run**, avoiding a full sampling pass with the lock silently inactive. |

### Output Ports

| Port | Type | Description |
|---|---|---|
| `model` | MODEL | Sampling model; connect to KSampler's model. On the Krea 2 in-context branch, if the upstream model has no in-context patch yet, this node adds it automatically; otherwise returned unchanged. |
| `tile_positives` | CONDITIONING list | One per tile. Connect to KSampler's positive. |
| `tile_negatives` | CONDITIONING list | One per tile. Connect to KSampler's negative. |
| `tile_latents` | LATENT list | One per tile. Connect to KSampler's latent_image. |

### Parameters

| Parameter | Type | Default | Description | Notes |
|---|---|---|---|---|
| `reference_mode` | BOOLEAN | False | ON = Krea 2 identity-edit (in-context layout); OFF = conditioning (auto-grounded depending on tokenizer capability). | **Only works with Krea 2**. When ON, non-Krea 2 falls back to conditioning with a warning. When ON, connect this node's `model` output straight to KSampler (the in-context patch is applied automatically); you only need to insert `Krea2 In-Context Patch` manually if you insist on using the original MODEL. |
| `grounding_px` | INT 0–4096 (step 32) | 768 | Vision-slot size. | Same semantics as TiledUpscale. |

### Branch Behavior (automatic)

| Condition | Branch |
|---|---|
| `reference_mode=True` and Krea 2 | in_context (attaches `reference_latents + index_timestep_zero`) |
| `reference_mode=False` AND prompt non-empty AND tokenizer handles images AND **not** FLUX.2 | grounded |
| otherwise | generic reference-latents |

### Required Patches

**This node outputs `model`**. On the Krea 2 in-context branch, if the upstream model has no in-context patch yet, this node applies it automatically (equivalent to `ElementKrea2InContextPatch`); other branches return the model unchanged and never double-patch.

| Which branch | Required patch |
|---|---|
| generic reference-latents / grounded | `ElementStrengthSchedule`: insert between this node's `model` output and KSampler to weaken references step by step |
| Krea 2 in-context | the in-context patch is already applied by this node; optionally chain `ElementStrengthSchedule` after the `model` output for per-step reference strength |

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
| `IMAGE` | IMAGE | Merged result. Size = `round(source × Split's U-upscale)`; at `U-upscale = 1.0` it matches the original image size exactly. |

### Parameters

| Parameter | Type | Default | Description | Notes |
|---|---|---|---|---|
| `feather` | FLOAT 0–1 | 1.0 | Feathering strength. | Between two hand-drawn tiles **of the same layer** the transition sits in the repaintable region between their exclusion bands, width = that region × feather. On a side facing the background (`gap / fallback / unified`) it stops outward at the tile's own exclusion boundary (the weight reaches exactly 0 there, so unedited reference pixels in the locked band never enter the composite semi-transparently) and inward at the overlap midline, with the 50% point always on the hand-drawn box edge. **Rounded corners** follow the same rule as a radial ramp: center = the drawn arc's center, 50% on the arc of radius `r`, reaching 0 at `r + repaintable width` — so the corner is controlled by this node's `feather`; Split's corner feather only shapes the sampling lock. Background tiles drop no band, so their transition spans the whole overlap. `0` = hard switch at the transition's midline (the midline between the locked bands for same-layer pairs, exactly the box edge when facing the background), `0.5` = only its middle half blends, `1` = the whole repaintable region blends. Weights are complementary and never blend with the underlay. Uses smoothstep, not linear. |
| `color_match_strength` | FLOAT 0–1 | 0.0 | Color matching strength. | **Only triggers when > 0**, and only truly works when **`tile_sources` is also connected**. Statistics use **low-weight edge pixels** (< 0.8) to avoid washing out center content. |
| `color_protection` | BOOLEAN | False | Hue/Color Protection | When enabled, each tile is compared against its source content; color-matching adjustments are applied only to pixels that have been intentionally recolored. This preserves manual hue or color modifications while ensuring that exposure corrections remain effective. When disabled, color matching is applied to each tile as usual. |
| `high_low_freq` | ​​BOOLEAN | False | High/Low-Frequency Blending (applies only to feathered regions) | Uses "feathered" weighting for low-frequency components to ensure smooth transitions, while employing "competitive" weighting for high-frequency components to preserve fine details—preventing details in overlapping areas from being lost due to averaging. |

### Merge Algorithm

1. Underlay: `original_image` scaled to `work_w × work_h`.
2. Hand-drawn tiles drop Split's locked band via the grid's `edit_mask` (only the band itself, `≤ overlap/2`). Background tiles (`gap / fallback / unified`) drop nothing: their band holds another enhancement of the same source, so excluding it would push the weight sum below 1 and let the underlay through.
3. A hand-drawn tile's feather transition sits between the two locked bands only **between neighbours of the same layer** (`edit_bands` from the grid plus the facing neighbour's), and its width is `overlap` minus both locked bands (`grid["overlap"]` is the only overlap source: in manual mode every sub-tile is expanded by just `overlap/2`, and same-layer sub-tiles no longer add a second overlap), so 50% lands in the middle of the two tiles' overlap. A side facing the background also starts at this tile's exclusion boundary, so the ramp never enters the locked band while the 50% point stays on the hand-drawn box edge. Weight is exactly 0 at that boundary, so the hard exclusion leaves no seam.
4. Rounded corners use a radial ramp with the same rule: 50% on the drawn arc, 0 at `r + repaintable width`, 1 further inward, width scaled by `feather`; `feather = 0` cuts hard along the drawn arc. The expanded outline (`r + overlap/2`) no longer cuts the corner, so corners are not trimmed along the expanded arc or a square boundary.
5. When the hand-drawn box is inset from the canvas edge (its `layer_region` is inset from the tile edge), that tile edge gets its transition back even though it touches the canvas: width = 2 × inset, 50% at the box edge. No inset (the box itself touches the edge) → no transition.
6. Foreground (`kind = layer`) accumulates into fg canvas; background (`gap / fallback / unified`) accumulates into bg canvas.
7. Normalize then composite: `base = bg * bg_alpha + original * (1 - bg_alpha)` → `merged = fg * fg_alpha + base * (1 - fg_alpha)`.
8. Crop to `img_h × img_w` (oversampled regions are trimmed).

### Notes

- **Without `tile_sources`, `color_match_strength` is completely ineffective**.
- **When Split is in manual mode and `fill gaps` is not active**, not connecting `original_image` causes uncovered missing areas to show black underlay.
- `feather = 0` switches hard at the repaintable region's midline (not at the tile edge).
- With Split's `edge mask` on, only **neighbours of the same layer** narrow the transition to the region between their locked bands; a hand-drawn tile's side facing the background starts at its own exclusion boundary with the box edge as the 50% point, so even a large `lock` cannot eat the box border or its rounded corners, and the exclusion boundary leaves no hard step. Within one layer, wider bands leave a narrower middle (per-side transition width = `overlap/2 − this tile's band`; a large `lock` can leave only a hard seam between two tiles of that layer), so widen the transition by lowering `lock`. Background tiles are unaffected and always blend across the whole overlap.
- **Rounded corners depend only on this node's `feather`**: Split's `corner feather` merely shapes the sampling lock (ramping inward), and Merge no longer trims the corner along an expanded outline or a square boundary.
- You can use the [Smart merge images node](https://github.com/supElement/ComfyUI_Element_easy) from another extension as a replacement; it provides better results for color and distortion correction. When connected to Split's `grid`, it reuses this node's alpha / feather geometry (centred ramps, same-layer transitions between lock bands, corner ramps around the drawn arc); hand-drawn tiles exclude their locked band and background tiles do not.
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
| Split version + Krea 2 + Encode's `reference_mode = ON` | **Not needed** (use Encode's `model` output; the patch is applied automatically). Only required if you insist on feeding the original MODEL to KSampler |
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

- **Must be connected to KSampler's model input**. Split doesn't output MODEL; connecting it after Split does nothing. In the split version, insert it between Encode's `model` output and KSampler.
- **Don't use it together with TiledUpscale**: the all-in-one already attaches the same-named patch internally; re-attaching gets stripped by `_strip_owned_patches` and re-added, which is redundant.
- For decimal separators, **use the English period `0.8`**; Chinese periods will fail to parse.
- Can accept the `strength_string` output from [Element_listGraph_Curve](https://github.com/supElement/ComfyUI_Element_easy).
- Lower parameter values ​​increase the likelihood of "drift." To ensure consistency, use higher values ​​for the initial steps—ideally, set the value to 1 for the first few steps. For subsequent steps, the value can be adjusted downward depending on the specific model being used; this approach is more effective for correcting imperfections. This parameter is independent of the `denoise` setting and does not affect vision- or grounding-based references. It is not applicable to models not designed for image editing (e.g., SDXL, Qwen-Image-2512, etc.).
