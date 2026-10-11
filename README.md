<div align="center">

[![中文](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI-Element_TileUpscale

一个专为**参考潜空间编辑模型**设计的分块精修 / 放大节点。它让 FLUX.2 klein、Qwen-Image 2.1 / Edit 2509 / 2511、Krea 2 这类模型在分块放大时，分块参考 + 邻居上下文 + 频率分离融合三层机制共同保证跨块一致性，重绘强度可以放心开到 **0.7–1.0**，同时保持原图的身份、结构与纹理一致——这是传统 img2img 分块放大做不到的。
- Krea2 需要 [krea2_identity_edit](https://huggingface.co/conradlocke/krea2-identity-edit/tree/main) lora 。  

---

## 为什么需要它

几乎所有传统的分块放大节点（TTP、Ultimate SD Upscale 及其衍生）走的都是 **img2img 路径**：把低分辨率原图加噪后喂给模型，让模型去噪重建。

这条路径有一个**结构性缺陷**——"重绘力度"和"特征保留"共用同一个通道，也就是降噪值：

| 降噪值 | 结果 |
|---|---|
| 低（≤0.4） | 原图信号保留多，特征勉强对得上，但模型重绘自由度不足 → **模糊、塑料感、细节缺失** |
| 高（≥0.5） | 细节和锐度上来了，但原图信号被噪声淹没 → **身份漂移、结构改变、跨块重影** |

在 img2img 范式下这个矛盾**无法调和**，因为你只有一个信息通道。所以传统节点的降噪值实际上被锁死在 0.1–0.5 这个狭窄窗口里，放大的上限也就卡在那里。更麻烦的是，如果用的是**非图像编辑模型**（普通 SDXL / Flux t2i），它根本没有"参考图"概念，一致性只能靠降噪值硬扛，结果更不可控。

---

## 它做了什么：把"重绘"和"保特征"拆成两条独立通道

这个节点不再依赖加噪输入来传递原图信息，而是把原图作为**独立的参考信号**注入模型——参考信号是**干净的**，不参与加噪，也不随降噪值衰减。

| 通道 | 作用 | 是否受降噪值影响 |
|---|---|---|
| 目标 latent（加噪输入） | 决定这一块要生成什么 | 是，降噪越高重绘越充分 |
| 参考 latent / in-context 参考 | 告诉模型"原图长这样" | **否**，始终以干净 token 存在 |

于是降噪值**只控制重绘力度，不再控制特征保留**：

- **降噪 0.7-1.0**：目标几乎可以从纯噪声重建（不同模型有差异），细节完全由模型生成；同时模型每一步都看着参考图，知道该生成什么内容、什么身份、什么纹理。**清晰度和一致性同时拿到。**
- **分块参考**：防止分块各自为政；给每一块精确的局部对应，防止细节错位。让高降噪下的**跨块一致性**也成立。
- **邻居上下文**：已生成的相邻块会以全强度反馈给下一块作为输入上下文，让模型把它当作"要延续的内容"，而不是靠重叠区模糊混合去掩盖接缝。
- **低频 / 高频分离融合**：拼接时低频（曝光、色彩）走平滑权重，高频（细节、纹理）走近乎独占的权重，避免轮廓重影和接缝模糊。
- **自动对齐**：节点会从模型中读取 VAE 下采样率和 DiT patch size，把分块尺寸对齐到模型的 token 像素网格，并按需自动规划分块几何。

---

## 什么时候该选它

**选它，如果你：**

- 用 FLUX.2 klein / Qwen-Image 2.1 / Edit 2509 / 2511 / Krea 2（+identity_edit）做放大或精修；
- 想要**明显更高的清晰度和细节**，而不是在 0.2–0.5 降噪下妥协；
- 需要**跨分块的身份、结构、纹理一致**，不要重影和接缝；
- 想要一个**单节点、开箱即用**的方案，而不是自己搭一整套分块工作流。

**它可能不适合你，如果你：**

- 需要语义 / 对象感知的可变尺寸分块；
- 需要逐节点精细控制整个分块流程；
- 使用的模型本身不具备任何参考潜空间能力，且你的目标只是轻微放大。

---

## 已知限制与缺点

这个节点用复杂度换效果，因此也有明确的边界。以下问题在使用前应当知悉：

### 1. 依赖特定的模型能力，通用性有限

- 节点的全部价值建立在**模型具备参考潜空间（reference latents）或 in-context 参考能力**之上。用 SDXL、普通 Flux t2i、Qwen-Image-2512 这类无参考能力的模型时，它会退化成普通的 tiled img2img——此时它**并不比 TTP 等工具更好**，甚至因为封装更重而显得笨拙。

### 2. 代码依赖逆向工程，稳定性脆弱

- 对模型的识别依赖**类名子串匹配**（`"flux" in type(dm).__name__.lower()`）、**tokenizer 签名反射**（`inspect.signature`）和 `vars()` 遍历。这些都不是稳定 API。
- Krea 2 的 in-context 路径是**手动重建前向传播**，直接调用 `dm.blocks`、`dm.pe_embedder`、`img_slice` 等内部结构。一旦上游模型实现变动（改名、改签名、改内部布局），节点很可能**静默出错或直接崩溃**。
- 这意味着它对 ComfyUI 本体和模型实现的更新**比较敏感**，长期维护成本高。

### 3. 性能与显存开销更大(在可接受范围内)

- 每个 tile 都要**独立编码参考 latent**；开启 grounding 时，每个 tile 还要**重新走一遍 text encoder 的 vision 编码**。分块越多，这部分开销线性增长。
- Krea 2 的 in-context 路径需要**手动跑完整的 DiT 前向**，token 序列是 `text + source + target` 三段拼接，注意力成本比原生长。
- tile 的 token 数增长时，注意力成本按**平方**增长，因此自动 tile 尺寸被限制在 1024–1536。

### 4. 参数敏感，需要一定调参经验

- `feather`、`overlap`、`grounding_px`、`denoise` 之间互相影响。默认值在多数情况下合理，但要拿到最佳效果通常需要按模型和图微调。
- 开 `clip` + `prompt` 时，`positive` / `negative` 输入会被**绕过**，改用逐 tile 重新编码的 conditioning。不熟悉这一点容易接错线。

### 5. 不支持语义 / 对象感知分块

- 分块是**规则的几何网格**，不理解图像内容。对构图复杂、主体跨块。
- 没有交互式分块编辑器，也没有内置的 tile 预览 / 手动重绘单块的工作流——它追求的是"一键出结果"，而不是"精细干预"。

### 6. 封装为单节点，灵活性受限

- 所有步骤（放大、分块、采样、融合）封装在一个节点里。想要**替换其中某一步**、插入自定义后处理、或把中间结果（`split_images` / `resampler_images`）拿去走别的分支，虽然节点有输出，但流程本身不易拆解重组。
- 相比之下，TTP 那样的模块化工具集更适合搭建非标准工作流。

### 7. 不保证对未适配模型的正确性

- 对节点未显式适配的模型家族，它**不会报错**，而是尽量走通用路径。但参考注入是否正确、token 对齐是否成立，取决于模型自身的实现——结果可能只是"能跑"，一致性未必有保证。

---

## 推荐降噪值与搭配

- 文本编码节点可统一使用 CLIP Text Encode (Prompt)，如果使用了图像编辑模型的专用文本编码节点，不要在文本编码节点中接入参考图。
- CFG取值应根据使用的模型设置。
- 编辑模型deniose取值范围0.6-1.0；非编辑模型应deniose取值不超过0.5。
- 编辑模型接上 `clip` 并填写 `prompt`，让每个 tile 以自身作为 vision 输入重新编码；Krea 2 identity-edit 需打开 `reference_mode`。
- 若效果不理想，优先调整 `grounding_px`、`feather` 和 `overlap`，而不是降低 `denoise`。

推荐 **Qwen-Image 2.1** 或 **Krea 2 + identity_edit**）。

| 模型 | 推荐降噪值 | 参考注入方式 | 一致性 | 说明 |
|---|---|---|---|---|
| **Qwen-Image 2.1** | **1.0** | 参考 latent 精确插在 vision token 位置（image_slots），tile 保持原尺寸不 resize | ★★★★★ | 参考信号最原生，多数情况下可直接拉满。细节完全由模型生成，同时身份 / 结构保持一致 |
| **Qwen-Image Edit 2509** | **1.0** | 384×384 vision bucket + reference latent 双通道 | ★★★★★ | 与 2511 同源，参考通道成熟，可放心拉满 |
| **Qwen-Image Edit 2511** | **1.0** | 384×384 vision bucket + reference latent 双通道 | ★★★★★ | 同 2509，推荐直接用 1.0 |
| **Krea 2 + krea2_identity_edit** | **0.7 – 0.8** | 手动重建 `[text｜source(frame=1)｜target(frame=0)]` in-context 前向 | ★★★★★ | 高降噪下一致性最好，但有时会出现脏点 **不建议拉满**；0.7–0.8 是清晰度与稳定的最佳平衡点 |
| **FLUX.2 klein** | **0.7 – 0.8** | 仅 reference-latent 通道（text encoder 为纯文本，无 vision grounding） | ★★★ | 缺少 vision 通道，参考信号弱一档，容易脏、偏色, 可使用[Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 做色彩或变形校正，搭配防偏移类lora使用时可提升至★★★★ |
| **Qwen-Image-2512/SDXL系列** | **0.2 – 0.5** | 无 | ★ | 非编辑模型无一致性锚定，即使接入ipadapter等方法，效果依然有限 |

以上测试数量有限，存在片面和主观因素，还是要根据不同情况来调整参数

---

## 📦 安装

### 方式一：手动安装（Manual Installation）

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI-Element_TileUpscale.git
```

### 方式二：通过 Manager 安装（Install using Manager）

在 ComfyUI Manager 中搜索 `ComfyUI-Element_TileUpscale`，点击 Install。

依赖即 ComfyUI 本体，无额外安装项。

---

## 参数说明

### 基本

| 参数 | 说明 |
|---|---|
| **image** | 要放大的源图 |
| **upscale_by** | 放大倍率，1.0 = 只做重采样精修不放大 |
| **tile_size** | 每块的重采样分辨率。0 = 自动（长边约 4 块，1024–1536）。块越大总块数越少、单块耗时越长 |
| **overlap** | 相邻切片间的重叠量。0 = 自动：切片尺寸的 15%，上限为 256px（取两者中的较小值）。任何非零值将按原样使用，不设上限。数值越高，接缝越平滑，但计算开销越大（开销随重叠量呈二次方增长）。|
| **feather** | 块间融合的羽化强度，1.0 = 最大平滑，一般不动 |
| **steps / cfg / sampler / scheduler / denoise / seed** | 与 KSampler 同义，直接套用该模型的官方推荐值 |
| **positive / negative** | 标准 conditioning 输入；接了 clip 并填写 prompt（或打开 reference_mode）时被绕过 |

### 可选（参考图模式组）

| 参数 | 说明 |
|---|---|
| **reference_mode** | 开 = Krea 2 identity-edit 专用 in-context 路径；关 = 通用 conditioning 路径。 |
| clip | 接入并填写 prompt 后，每个分块连同自身像素一起重新编码（官方参考图模式）。Qwen 2.1 / Edit 2509 / 2511 必接；Krea 2 identity-edit（reference_mode 开）必接；FLUX.2 klein 不需要接——其文本编码器为纯文本，无 vision 槽，接入无效 |
| **prompt** | 与 clip 配套的逐块指令文本，例如： enhance quality preserving all the details |
| redo_first_tile | 在所有其他图块处理完成后，再次重新生成第一个图块，以便它能将已完成的相邻图块作为上下文参考。此操作仅在存在多个图块时适用。 |
| grounding_px | 逐块文本编码的 vision 输入尺寸，默认 768，含义随模型不同：Edit 2509/2511 为 area 桶，0 或 ≥384 一律使用官方 384×384；Qwen 2.1 忽略此参数（锁定为分块自身尺寸）；Krea 2 为视觉输入最长边上限，0 = 分块自身尺寸 |
| reference_strength | “每图块参考强度” 决定如何锚定原始图像。设定为 1.0 时表示完整参考。较低的数值会降低分辨率后构建参考，这样既保留了构图和色彩信息，又放宽了对轮廓的限制，从而允许模型重新绘制原本有缺陷的形状。 |

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
显存不够？ 减小 tile_size（分块更多、单块更小）。Edit 2509/2511 可额外降低 grounding_px；Qwen 2.1 的 grounding_px 不参与计算，只能减小 tile_size。

**SDXL 等普通模型能用吗？**
能。节点会自动退化为普通分块 img2img，参考 latent 被模型忽略，用 denoise 控制细节强度（0.2 – 0.5 起试）。

---

