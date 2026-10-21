<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI-Element_TileEdit
  <sub> Original Name: ComfyUI-Element_TileUpscale </sub>

A comfyUI extension designed for image editing models, featuring tiled refinement/editing/upscaling. It enables models like Qwen-Image 2.1 / Edit 2509 / 2511, FLUX.3/FLUX.2 klein, Krea2 to maintain cross-tile consistency during upscaled segmentation through a three-layer mechanism: tiled reference ≫ frequency separation fusion > neighbor context. This allows denoise redraw strength to be confidently set to 0.7–1.0 while preserving the identity, structure, and texture of the original image. The reference strength per sampling step can be customized to control the balance between repair strength and consistency. It retains some capabilities of image editing models (limited to single images). This is something traditional img2img tiled upscaling cannot achieve. It replaces Mask with spatial segmentation for multi-region, multi-instruction image editing. The manual segmentation mode (by Manual) provides true "tile-level semantic isolation" and pixel-level control: you can precisely assign tiles to "white shirt" and write "change to blue shirt" only in the corresponding Layer's Prompt. Meanwhile, the background, hair, and table not selected are automatically included in the Gap fill (gap filling) area, which does not receive the shirt-changing instructions (or only receives basic global quality enhancement instructions), thus perfectly maintaining their original state. Additionally, each manually segmented layer can have its own upscaling size, giving you higher resolution and detail expression when editing or redrawing core areas. Manual segmentation (CTRL + left-click drag to add tiles).

- Krea2 requires [krea2_identity_edit](https://huggingface.co/conradlocke/krea2-identity-edit/tree/main) lora.

<img width="2290" height="1123" alt="image" src="https://github.com/user-attachments/assets/ebb876d9-3bc6-436e-a87e-938b7dbf350a" />

---

<img width="2379" height="1093" alt="image" src="https://github.com/user-attachments/assets/1464d90d-e044-4423-9d27-7fa2a581a66e" />


## Node Overview

| Node | Function |
|---|---|
| **Element Tiled Upscale** | Integrated node: Upscale → Tile → Per-tile reference redraw → Frequency separation fusion, one-click output |
| **Element Tile Split** | Pure geometric tiling + image source input, no model dependency. The backend recalculates geometry in full during execution, while the frontend UI only serves as a preview. |
| **Element Tile Encode** | Encodes each tile image / prompt into per-tile LATENT + CONDITIONING, leaving sampling to ComfyUI's native KSampler |
| **Element Tile Merge** | Merges the list of images decoded per tile using the same low-frequency/high-frequency separation + feathering weights, optionally with per-tile color matching |
| **Element Krea2 In-Context Patch** | Model patch: Allows Krea2 to consume the reference latent hangs on conditioning with an in-context layout for identity-edit |
| **Element Strength Schedule** | Model patch: Attaches a gradual reference_strength schedule to the model, used by ElementTileSplit + native KSampler workflows, KSampler's model input must connect to the output of this node. The integrated node includes the same schedule and ignores upstream attachments; single-node workflows directly use its built-in parameters. The i-th value affects the i-th sampling step (starting from 0); if fewer than steps, repeat the last value; if more, truncate; a single value = constant. 1.0 = full reference; 0 = no reference for that step; pure 0/1 list uses single multiplicative power, non-integer values double forward for steps ≠ 1. Lower parameter settings lead to more drift; to ensure consistency, the first few steps should have high values, recommend setting the first few steps to 1, and gradually reduce the values for the later steps based on the model used, which is more beneficial for repairing flaws. This is ineffective for non-image editing models like sdxl, Qwen image 2512, etc. |

---

## Why It's Needed

Almost all traditional tiled upscaling nodes (TTP, Ultimate SD Upscale and its derivatives) follow the **img2img path**: adding noise to the low-resolution original and feeding it to the model for denoising and reconstruction.
This path has a **structural flaw**—"redraw strength" and "feature retention" share the same channel, which is the denoising value: low (≤0.4), more original signal is retained, features barely match but the model has insufficient redraw freedom → **blur, plastic feel, missing details**; high (≥0.5), details and sharpness improve, but the original signal is drowned out by noise → **identity shift, structural change, cross-tile ghosting**.
Under the img2img paradigm, this contradiction **cannot be reconciled**, as you only have one information channel. Therefore, the denoising value of traditional nodes is essentially locked in the narrow window of 0.1–0.5, and the upper limit of upscaling is also stuck there.

Element_TileEdit splits "redraw" and "feature retention" into two independent channels, injecting the original image as an independent reference signal into the model. The reference signal does not participate in noise addition and does not decay with denoising value. The denoising value **only controls redraw strength, no longer controls feature retention**

- **Denoise 0.7-1.0**: The target can almost be reconstructed from pure noise (with variations across models), details are completely generated by the model; at the same time, the model sees the reference image every step, knowing what content, identity, and texture to generate. **Clarity and consistency are achieved simultaneously.**
- **Tiled Reference**: Prevents tiles from acting independently; provides precise local correspondences to prevent detail misalignment. Allows **cross-tile consistency** even under high denoising.
- **Low-frequency / High-frequency Separation Fusion**: During merging, low-frequency (exposure, color) follows smooth weights, while high-frequency (details, texture) dominates nearly exclusively, avoiding contour ghosting and seam blurring.
- **Neighbor Context**: Already generated adjacent tiles feed back as input context with full intensity to the next tile, making the model treat them as "content to be continued," rather than relying on blurred overlaps to hide seams. (For reference-anchored models, neighbor context mainly softens the microscopic texture of the seam band when denoise < 1; structural and identity consistency is handled by the shared original image reference, which is why redo_first_tile is almost ineffective for editing models.)

---

## Known Limitations and Disadvantages

### 1. Relies on Specific Model Capabilities, Unfriendly to Non-Image Editing Models

- The entire value of the node is built on the **model's ability to reference latent space**. Using models without this capability like SDXL, standard Flux t2i, Qwen-Image-2512, etc., it degrades into a normal tiled img2img—**at this point, it is no better than tools like TTP.**

### 2. Model Recognition Depends on Reverse Engineering, Potential for Inconsistency Across ComfyUI Versions

- Model recognition relies on **substring matching of class names** (`"flux" in type(dm).__name__.lower()`), **tokenizer signature reflection** (`inspect.signature`), and `vars()` traversal. None of these are stable APIs.
- Krea2's in-context path involves **manual reconstruction of forward propagation**, directly calling `dm.blocks`, `dm.pe_embedder`, `img_slice`, etc., internal structures; the Patch node also depends on core's `extra_conds` object-patch and DIFFUSION_MODEL wrapper mechanisms. Once the upstream model implementation changes (renaming, signature change, internal layout change), the node may **silently fail or crash directly.**

### 3. Higher Performance and Memory Usage (Within Acceptable Limits)

- Each tile requires **independent encoding of reference latent**; when grounding is enabled, each tile also needs to **go through the text encoder's vision encoding** again. The overhead grows linearly with the number of tiles.
- Krea2's in-context path requires running the full DiT forward pass, with token sequences `text + source + target` three-section concatenated, attention cost higher than the original.
- As the number of tokens per tile increases, attention cost grows **quadratically**, thus automatic tile size is limited to 1024–1536. The training resolution upper limit for identity-edit LoRA is about 2MP, and tiles should not exceed 2MP (1536²≈2.36MP is already over the limit, 1408²≈1.98MP is close to the upper limit).

### 4. Does Not Guarantee Correctness for Unadapted Models

---
## Recommended Denoising Values and Combinations

- Text encoding nodes can uniformly use CLIP Text Encode (Prompt). If a dedicated text encoding node for image editing models is used, do not connect reference images to the text encoding node.
- CFG values should be set according to the model being used.
- For editing models, the denoise value range is 0.6–1.0; for non-editing models, the denoise value should not exceed 0.5.
- For editing models, connect `clip` and fill in `prompt` to re-encode each tile with itself as the vision input; enable `reference_mode` for Krea2 identity-edit.
- If the results are not ideal, prioritize adjusting `grounding_px`, `feather`, and `overlap` rather than reducing `denoise`.

| Model | Recommended Denoising Value | Reference Injection Method | Consistency | Description |
|---|---|---|---|---|
| **Qwen-Image 2.1** | **1.0** | Reference latent precisely inserted at vision token positions (image_slots), tile remains original size without resize | ★★★★★ | Reference signal is most native, can be pushed to the maximum in most cases. Details are fully generated by the model, while identity and structure remain consistent |
| **Qwen-Image Edit 2511** | **1.0** | 384×384 vision bucket + reference latent dual channel | ★★★★ | Same as 2509, recommended to use 1.0 directly, lacks details |
| **Qwen-Image Edit 2509** | **1.0** | 384×384 vision bucket + reference latent dual channel | ★★★★ | Same source as 2511, reference channel is mature, can be pushed to the maximum, lacks details |
| **FLUX.2 klein base** | **0.7 – 1.0** | Only reference-latent channel (text encoder is pure text, no vision grounding) | ★★★★ | Lacks vision channel, some oily faces, color bias, can use [Smart merge images node](https://github.com/supElement/ComfyUI_Element_easy) for color or deformation correction, can be improved to ★★★★★☆ when paired with anti-shift LoRA |
| **Krea2 + krea2_identity_edit** | **0.7 – 0.8** | Manually rebuild `[text｜source(frame=1)｜target(frame=0)]` in-context forward | ★★★☆ | Best color consistency under high denoising, but sometimes dirty spots may appear, 0.7–0.8 is the best balance between clarity and stability |
| **FLUX.3 klein turbo** | **0.7 – 0.8** | text encoder is only text effective, no vision grounding | ★★★ | Lacks details, prone to deformation (may be an issue with comfyUI implementation), advantage is not easily dirty |
| **FLUX.2 klein turbo** | **0.7 – 0.8** | Only reference-latent channel (text encoder is pure text, no vision grounding) | ★★★ | Lacks vision channel, reference signal is one level weaker, very easy to get dirty or color biased, can use [Smart merge images node](https://github.com/supElement/ComfyUI_Element_easy) for color or deformation correction, can be improved to ★★★☆ when paired with anti-shift LoRA |
| **Qwen-Image-2512/SDXL series** | **0.2 – 0.5** | None | ★ | Non-editing models lack consistency anchoring, even with methods like ipadapter, the effect is still limited |

- Model combination strategy: It is recommended to use non-distilled models like base or raw, paired with turbo LoRA (weight about 0.8, adjust according to the specific model). This can balance the high-quality detail restoration of the base model with the generation speed of the distilled model, and effectively avoid high-frequency artifacts (dirty spots) that are easily amplified when directly using pure turbo models.
- Reference strength curve (reference_strength): It is recommended to use an "early high, late low" tail relaxation curve. For example, when the total steps are 4, you can fill in 1.0, 1.0, 0.9, 0.8. Since turbo models have very few steps, the strength of each step greatly affects the result. It is recommended that the reference strength throughout should not be below 0.7 (otherwise it is easy to cause image drift); if feature drift occurs in actual use, the curve can be directly flattened to all 1.0.
- Krea2 model tuning: It is recommended to leave the negative_prompt completely empty. This is because krea2's identity-edit uses an unconditional distribution of "the same image + empty text" during training, leaving it empty can perfectly align with the training distribution; and when cfg=1.0, the node can also save the encoding time of negative prompt words.

  The above tests are limited in number and have biases and subjective factors, so parameters still need to be adjusted according to different situations. Of course, you can also improve the results by using enhanced LoRA or fine-tuned models, but this is not within the scope of the test.  
---

## Global Prompt (Non by manual mode)

In non-manual block mode (e.g., integrated node, by count, by size), this node will apply the same global prompt to every text encoding block. Since each block only covers a small local area of the original image, the specific content of this area is completely determined by the reference image (pixel features), while the prompt only guides the model "how to process" this content.

Therefore, the global prompt must follow the core principle of being "generally applicable to all blocks":

### ✅ Should Write
1. Operation instructions —— Tell the model the core task of this sampling (which applies to each block):
  - sharpen and refine details, preserve identity and structure, clean upscale  
  - (Sharpen and refine details, preserve identity and structure, cleanly upscale)

2. Global style / lighting / tone —— The atmosphere shared by the entire image (which also applies to each block):
  - cool blue ambient lighting, warm orange rim light, cinematic tone, photorealistic  
  - (Cool blue ambient light, warm orange rim light, cinematic tone, photorealistic)

### ❌ Should Not Write
  - Specific object, character, or scene nouns (e.g., woman at a computer, glasses, monitor, coffee cup).
  - Unless you can be sure that the prompt describes object elements that apply to every block in the image.

### 💡 Recommended Template
  - sharpen and refine, preserve identity and structure, {global tone/lighting/style}, {quality terms}
---

## 📦 Installation

### Method 1: Manual Installation (Manual Installation)
      cd ./ComfyUI/custom_nodes
      git clone https://github.com/supElement/ComfyUI-Element_TileEdit.git

### Method 2: Install using Manager (Install using Manager)

Search for `ComfyUI-Element_TileEdit` in ComfyUI Manager and click Install.

Dependencies are the ComfyUI itself, no additional installations required.
## Parameter Details: [Chinese](md/Description.md) | [English](md/Description_en.md)

---

## Common Questions

**How to use?**  
- Workflows are already provided in the workflow folder.  

**How to split manually?**

- You must use the manual mode in the Element Tile Split node. Hold down the CTRL key + left mouse button on the canvas, drag to add segments, and you can add them consecutively.
- Each added segment will add a corresponding layer in the right panel. Each layer can set its prompt separately in the layer prompt hint below.
- You can delete unnecessary segments (layers) by pressing delete, or you can click the delete button on the corresponding layer with the mouse to delete the segment.

**Insufficient VRAM?**  
- Reduce tile_size (more segments, smaller each). Edit 2509/2511 can additionally reduce grounding_px; Qwen 2.1's grounding_px is not involved in the calculation, only tile_size can be reduced.

**Can SDXL and other normal models be used?**  
- Yes. The node will automatically degrade to normal img2img segmentation. Refer to latent being ignored by the model, using denoise to control detail strength (start with 0.2 – 0.5).

**Do you need to connect a Patch node when using Krea 2 with a single node?**  
- `reference_mode` If it's on, you don't need to (it has an internal self-built mechanism of the same type, and it will be automatically ignored if connected); if it's off (conditioning path), **it must be connected**, otherwise the reference is silently discarded by the core. See the above description of the Patch node.