<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI-Element TileUpscale

An integrated tiled refine/upscale node, designed specifically for **reference image editing models**: Qwen Image 2.1, FLUX.2 klein, Krea2, etc. Krea2 requires krea2_identity_edit lora.

After upscaling the image, it is cut into overlapping tiles, each of which is regenerated at full resolution, and then seamlessly merged. It can bring in realistic new details while maintaining the original content through the reference image embedded in each tile.

---

## Installation

Place this folder into `ComfyUI/custom_nodes/`, restart ComfyUI. The node can be found by searching `Element Tiled Upscale` in the search bar (Category: Element TileUpscale).

Dependencies are just the ComfyUI itself, no additional installations required.

---

## Recommended Configurations for Each Model

Text encoding nodes can uniformly use CLIP Text Encode (Prompt). If you use a dedicated text encoding node for image editing models, do not connect the reference image to the text encoding node.

The CFG value should be set according to the model being used.

For editing models, deniose should be in the range of 0.6-1.0; for non-editing models, deniose should not exceed 0.5.

Editing models require a clip input port, and the prompt on the node should contain content.

---

## Parameter Description

### Basic

| Parameter | Description |
|---|---|
| **image** | The source image to upscale |
| **upscale_by** | The upscaling factor, 1.0 = only do resampling refinement without upscaling |
| **tile_size** | The resampling resolution of each tile. 0 = automatic (approximately 4 tiles along the longer edge, 1024–1536). Larger tiles result in fewer total tiles and longer processing time per tile |
| **overlap** | The overlapping pixels between adjacent tiles. 0 = automatic (30% of the tile size). Adjust it up if there are seam anomalies |
| **feather** | The feathering strength for blending between tiles, 1.0 = maximum smoothness, generally leave unchanged |
| **steps / cfg / sampler / scheduler / denoise / seed** | Same as KSampler, directly use the official recommended values for the model |
| **positive / negative** | Standard conditioning input; bypassed when clip is connected |

### Optional (Reference Image Mode Group)

| Parameter | Description |
|---|---|
| **reference_mode** | On = Krea 2 identity-edit specific in-context path; Off = general conditioning path. |
| **clip** | After connecting, each tile will be re-encoded along with its own pixels (official reference image mode). Qwen 2.1 / FLUX.2 klein / Krea 2 style-ref must be connected |
| **prompt** | The per-tile instruction text paired with clip |
| **grounding_px** | The bucket size for per-tile reference (area-based, a multiple of 32), default 1024. Larger values resemble the original image but are slower; 0 = use the tile's own size |

### Output

| Port | Content |
|---|---|
| **image** | The final merged full image (main output) |
| **split_images** | The list of source images after splitting (for debugging) |
| **resampler_images** | The list of generation results for each tile (for debugging, can check quality per tile) |

You can use the [Smart merge images node](https://github.com/supElement/ComfyUI_Element_easy) from my other extension for color or distortion correction.

---

When cfg = 1.0, the node will automatically skip useless negative text encoding, halving the encoding phase time.

---

## Common Issues

**Not enough VRAM?**
Reduce `tile_size` (more tiles, smaller per-tile size), or lower `grounding_px`. Model loading/unloading logs (Requested to load ...) are normal VRAM management behavior in ComfyUI and require no action.

**Can SDXL and other regular models be used?**
Yes. The node will automatically degrade to regular tiled img2img, and the reference latent will be ignored by the model, with denoise controlling detail strength (start with 0.4–0.7).

---

## Working Principle (Simplified)

Upscale source image → Align tiles by tokens (with overlap between tiles) → Each tile is resampled with "itself as reference image + already generated neighboring content as context" → Two-layer weighted blending (low-frequency to preserve exposure, high-frequency to preserve details) → Run the first tile one more time at the end to complete the context → Output the full image.