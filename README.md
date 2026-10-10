# ComfyUI-Element TileUpscale

一体化分块放大（tiled refine/upscale）节点，专为**参考图编辑类模型**设计：Qwen Image 2.1、FLUX.2 klein、Krea2 等。Krea2需要krea2_identity_edit lora。

把图像放大后切成带重叠的分块，每个分块在全分辨率下重新生成，再无缝拼合。 它能带来真实的新细节，同时靠每块自带的参考图保持原图内容。

---

## 安装

将本文件夹放入 `ComfyUI/custom_nodes/`，重启 ComfyUI。节点在搜索栏输入 `Element Tiled Upscale` 即可找到（类别：Element TileUpscale）。

依赖即 ComfyUI 本体，无额外安装项。

---

## 各模型推荐配置

文本编码节点可统一使用 CLIP Text Encode (Prompt)，如果使用了图像编辑模型的专用文本编码节点，不要在文本编码节点中接入参考图。

CFG取值应根据使用的模型设置。

编辑模型deniose取值范围0.6-1.0；非编辑模型应deniose取值不超过0.5。

编辑模型需接 clip输入端口，节点上的 prompt 需要有内容。
---

## 参数说明

### 基本

| 参数 | 说明 |
|---|---|
| **image** | 要放大的源图 |
| **upscale_by** | 放大倍率，1.0 = 只做重采样精修不放大 |
| **tile_size** | 每块的重采样分辨率。0 = 自动（长边约 4 块，1024–1536）。块越大总块数越少、单块耗时越长 |
| **overlap** | 相邻块的重叠像素。0 = 自动（tile 的 30%）。接缝异常时优先调大它 |
| **feather** | 块间融合的羽化强度，1.0 = 最大平滑，一般不动 |
| **steps / cfg / sampler / scheduler / denoise / seed** | 与 KSampler 同义，直接套用该模型的官方推荐值 |
| **positive / negative** | 标准 conditioning 输入；接了 clip 时被绕过 |

### 可选（参考图模式组）

| 参数 | 说明 |
|---|---|
| **reference_mode** | 开 = Krea 2 identity-edit 专用 in-context 路径；关 = 通用 conditioning 路径。 |
| **clip** | 接入后每个分块会连同自身像素一起重新编码（官方参考图模式）。Qwen 2.1 / FLUX.2 klein / Krea 2 style-ref 必接 |
| **prompt** | 与 clip 配套的逐块指令文本 |
| **grounding_px** | 逐块参考的桶尺寸（面积基准，32 的倍数），默认 1024。调大更像原图、更慢；0 = 用分块自身尺寸 |

### 输出

| 端口 | 内容 |
|---|---|
| **image** | 最终拼合完成的整图（主输出） |
| **split_images** | 切块后的源图列表（调试用） |
| **resampler_images** | 每块的生成结果列表（调试用，可逐块检查质量） |

可以使用我的另一个扩展中的 [Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 做色彩或变形校正。

---

cfg = 1.0 时节点会自动跳过无用的负向文本编码，编码阶段耗时减半。

---

## 常见问题

**显存不够？**
减小 `tile_size`（分块更多、单块更小），或降低 `grounding_px`。模型加载/卸载日志（Requested to load ...）是 ComfyUI 正常的显存管理行为，无需处理。

**SDXL 等普通模型能用吗？**
能。节点会自动退化为普通分块 img2img，参考 latent 被模型忽略，用 denoise 控制细节强度（0.4–0.7 起试）。

---

## 工作原理（简述）

放大源图 → 按 token 对齐切块（块间带重叠）→ 每块以"自身为参考图 + 邻居已生成内容为上下文"重新采样 → 两层加权融合（低频保曝光、高频保细节）→ 首块最后重跑一次补齐上下文 → 输出整图。
