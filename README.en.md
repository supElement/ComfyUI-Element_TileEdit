<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI-Element TileUpscale

An integrated tiled refine/upscale node, specifically designed for **reference image editing models**: Qwen Image 2.1, FLUX.2 klein, Krea2, etc. Krea2 requires [krea2_identity_edit](https://huggingface.co/conradlocke/krea2-identity-edit/tree/main) lora.

After upscaling the image, it is cut into overlapping tiles. Each tile is then resampled at full resolution and seamlessly combined. It can add realistic new details while maintaining the original content through the reference image embedded in each tile.

---

## 📦 Installation

### Method 1: Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI-Element_TileUpscale.git
```

### Method 2: Install using Manager

Search for `ComfyUI-Element_TileUpscale` in the ComfyUI Manager and click Install.

Dependencies are the ComfyUIOntology itself, with no additional installations required.

---

## Recommended Configurations for Each Model

Text encoding nodes can uniformly use CLIP Text Encode (Prompt). If a model's dedicated text encoding node is used, do not connect the reference image to the text encoding node.

The CFG value should be set according to the model being used.

For editing models, the denoise value range is 0.6-1.0; for non-editing models, the denoise value should not exceed 0.5.

Editing models require a clip input port, and the prompt on the node must contain content.

---

## Parameter Description

### Basic

| Parameter | Description |
|---|---|
| **image** | The source image to upscale |
| **upscale_by** | Upscaling multiplier, 1.0 = only resampling refinement without actual upscaling |
| **tile_size** | The resampling resolution of each tile. 0 = automatic (approximately 4 tiles along the longer edge, 1024–1536). Larger tiles result in fewer total tiles and longer processing time per tile |
| **overlap** | Overlapping pixels between adjacent tiles. 0 = automatic (30% of the tile). Adjust this first if seams are abnormal |
| **feather** | Feathering strength for blending between tiles, 1.0 = maximum smoothness, generally unchanged |
| **steps / cfg / sampler / scheduler / denoise / seed** | Same as KSampler, directly use the official recommended values for the model |
| **positive / negative** | Standard conditioning input; bypassed when connected to clip |

### Optional (Reference Image Mode Group)

| Parameter | Description |
|---|---|
| **reference_mode** | On = Krea 2 identity-edit specific in-context path; Off = general conditioning path. |
| **clip** | After connection, each tile will be re-encoded along with its own pixels (official reference image mode). Qwen 2.1 / FLUX.2 klein / Krea 2 style-ref must be connected |
| **prompt** | Block-by-block instruction text paired with clip |
| **grounding_px** | Block-by-block reference bucket size (area-based, multiples of 32), default 1024. Larger values resemble the original image but are slower; 0 = use the tile's own dimensions |

### Output

| Port | Content |
|---|---|
| **image** | The final merged full image (main output) |
| **split_images** | List of source images after splitting (for debugging) |
| **resampler_images** | List of generation results for each tile (for debugging, can check quality per tile |

You can use the [Smart merge images node](https://github.com/supElement/ComfyUI_Element_easy) from another extension for color or deformation correction.

---

When cfg = 1.0, the node will automatically skip unnecessary negative text encoding, halving the encoding phase time.

---

## Common Issues

**Not enough VRAM?**
Reduce `tile_size` (more tiles, smaller per tile), or lower `grounding_px`. Model loading/unloading logs (Requested to load ...) are normal VRAM management behavior in ComfyUI and require no action.

**Can SDXL and other regular models be used?**
Yes. The node automatically degrades to a regular tiled img2img, with the reference latent ignored by the model, using denoise to control detail strength (start with 0.4–0.7).

---

## Working Principle (Simplified)

Upscale source image → Align tiles by tokens with overlap between them → Each tile is resampled with "itself as reference image + already generated neighboring content as context" → Two-layer weighted blending (low-frequency for exposure, high-frequency for details) → The first tile runs one more pass to complete the context → Output the full image.