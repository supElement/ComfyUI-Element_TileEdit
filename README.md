<div align="center">

[![中文](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI-Element_TileUpscale

一个专为图像编辑模型设计的分块精修 / 编辑 / 放大节点。它让 Qwen-Image 2.1 / Edit 2509 / 2511、FLUX.3/FLUX.2 klein、Krea2 这类模型在分块放大时，以 分块参考 ≫ 频率分离融合 > 邻居上下文 三层机制共同保证跨块一致性，denoise 重绘强度可以放心开到 0.7–1.0，同时保持原图的身份、结构与纹理一致；每采样步参考强度可以自定义，控制修复强度与一致性的平衡；延续部分图像编辑模型的能力（仅限单图）。这是传统 img2img 分块放大做不到的。

- Krea2 需要 [krea2_identity_edit](https://huggingface.co/conradlocke/krea2-identity-edit/tree/main) lora。

<img width="2290" height="1123" alt="image" src="https://github.com/user-attachments/assets/ebb876d9-3bc6-436e-a87e-938b7dbf350a" />

---

<img width="2291" height="857" alt="image" src="https://github.com/user-attachments/assets/9725c148-a2ea-41c0-bfe2-607ec64d3c04" />


## 节点一览

| 节点 | 作用 |
|---|---|
| **Element Tiled Upscale** | 一体化节点：放大 → 切块 → 逐块参考重绘 → 频率分离融合，一键出图 |
| **Element Tile Split** | 只做放大 + 切块 + 逐块编码，输出逐块 LATENT / CONDITIONING / 原图分块，采样交给 ComfyUI 原生 KSampler |
| **Element Tile Merge** | 把逐块解码后的 IMAGE 列表用同一套低频/高频分离 + 羽化权重拼回整图，可选逐块色彩匹配 |
| **Element Krea2 In-Context Patch** | 模型补丁：让 Krea2 以 identity-edit 的 in-context 布局消费挂在 conditioning 上的参考 latent |
| **Element Strength Schedule** | 模型补丁：把逐步 reference_strength 调度挂到模型上，供 ElementTileSplit + 原生 KSampler 工作流使用，KSampler 的 model 输入必须接本节点输出。一体化节点内置同款调度且会忽略上游挂载，单节点工作流直接用其自带参数即可。第 i 个值作用于第 i 个采样步（0 起）；少于 steps 重复末值，多于截取；单值 = 恒定。 1.0 = 完全参考；0 = 该步完全无参考；纯 0/1 列表单倍算力，含小数时值≠1 的步双前向。参数设置越低，越会出现漂移，为保证一致性，前几步数值设置要高，建议前几步设置为1，后面几步可根据所使用的模型适当降低，这种设置对修复瑕疵更有利。 对非图像编辑模型，如sdxl、Qwen image 2512等模型无效 |

---

## 为什么需要它

几乎所有传统的分块放大节点（TTP、Ultimate SD Upscale 及其衍生）走的都是 **img2img 路径**：把低分辨率原图加噪后喂给模型，让模型去噪重建。
这条路径有一个**结构性缺陷**——"重绘力度"和"特征保留"共用同一个通道，也就是降噪值：低（≤0.4），原图信号保留多，特征勉强对得上，但模型重绘自由度不足 → **模糊、塑料感、细节缺失**；高（≥0.5），细节和锐度上来了，但原图信号被噪声淹没 → **身份漂移、结构改变、跨块重影** 
在 img2img 范式下这个矛盾**无法调和**，因为你只有一个信息通道。所以传统节点的降噪值实际上被锁死在 0.1–0.5 这个狭窄窗口里，放大的上限也就卡在那里。

Element_TileUpscale 把"重绘"和"保特征"拆成两条独立通道，把原图作为独立的参考信号注入模型，参考信号不参与加噪，也不随降噪值衰减。 降噪值**只控制重绘力度，不再控制特征保留**

- **降噪 0.7-1.0**：目标几乎可以从纯噪声重建（不同模型有差异），细节完全由模型生成；同时模型每一步都看着参考图，知道该生成什么内容、什么身份、什么纹理。**清晰度和一致性同时拿到。**
- **分块参考**：防止分块各自为政；给每一块精确的局部对应，防止细节错位。让高降噪下的**跨块一致性**也成立。
- **低频 / 高频分离融合**：拼接时低频（曝光、色彩）走平滑权重，高频（细节、纹理）走近乎独占的权重，避免轮廓重影和接缝模糊。
- **邻居上下文**：已生成的相邻块会以全强度反馈给下一块作为输入上下文，让模型把它当作"要延续的内容"，而不是靠重叠区模糊混合去掩盖接缝。（对参考锚定模型而言，邻居上下文主要在 denoise < 1 时软化接缝带的微观纹理；结构与身份的一致性由共享的原图参考承担，这也是 redo_first_tile 对编辑模型几乎无效的原因。）

---

## 已知限制与缺点

### 1. 依赖特定的模型能力，对非图像编辑模型不友好

- 节点的全部价值建立在**模型具备参考潜空间参考能力**之上。用 SDXL、普通 Flux t2i、Qwen-Image-2512 这类无参考能力的模型时，它会退化成普通的 tiled img2img——此时它**并不比 TTP 等工具更好**。

### 2. 模型识别依赖逆向工程，可能存在 comfyUI 跨版本稳定性

- 对模型的识别依赖**类名子串匹配**（`"flux" in type(dm).__name__.lower()`）、**tokenizer 签名反射**（`inspect.signature`）和 `vars()` 遍历。这些都不是稳定 API。
- Krea2 的 in-context 路径是**手动重建前向传播**，直接调用 `dm.blocks`、`dm.pe_embedder`、`img_slice` 等内部结构；Patch 节点还依赖 core 的 `extra_conds` object-patch 与 DIFFUSION_MODEL wrapper 机制。一旦上游模型实现变动（改名、改签名、改内部布局），节点很可能**静默出错或直接崩溃**。

### 3. 性能与显存开销更大(在可接受范围内)

- 每个 tile 都要**独立编码参考 latent**；开启 grounding 时，每个 tile 还要**重新走一遍 text encoder 的 vision 编码**。分块越多，这部分开销线性增长。
- Krea2 的 in-context 路径需要跑完整的 DiT 前向，token 序列是 `text + source + target` 三段拼接，注意力成本比原生长。
- tile 的 token 数增长时，注意力成本按**平方**增长，因此自动 tile 尺寸被限制在 1024–1536。identity-edit LoRA 的训练分辨率上限约 2MP，tile 请尽量不超过 2MP（1536²≈2.36MP 已越界，1408²≈1.98MP 贴上限）。

### 4. 不保证对未适配模型的正确性

---

## 推荐降噪值与搭配

- 文本编码节点可统一使用 CLIP Text Encode (Prompt)，如果使用了图像编辑模型的专用文本编码节点，不要在文本编码节点中接入参考图。
- CFG 取值应根据使用的模型设置。
- 编辑模型 denoise 取值范围 0.6–1.0；非编辑模型 denoise 取值不超过 0.5。
- 编辑模型接上 `clip` 并填写 `prompt`，让每个 tile 以自身作为 vision 输入重新编码；Krea2 identity-edit 打开 `reference_mode`。
- 若效果不理想，优先调整 `grounding_px`、`feather` 和 `overlap`，而不是降低 `denoise`。

| 模型 | 推荐降噪值 | 参考注入方式 | 一致性 | 说明 |
|---|---|---|---|---|
| **Qwen-Image 2.1** | **1.0** | 参考 latent 精确插在 vision token 位置（image_slots），tile 保持原尺寸不 resize | ★★★★★ | 参考信号最原生，多数情况下可直接拉满。细节完全由模型生成，同时身份 / 结构保持一致 |
| **Qwen-Image Edit 2509** | **1.0** | 384×384 vision bucket + reference latent 双通道 | ★★★★☆ | 与 2511 同源，参考通道成熟，可放心拉满 |
| **Qwen-Image Edit 2511** | **1.0** | 384×384 vision bucket + reference latent 双通道 | ★★★★☆ | 同 2509，推荐直接用 1.0 |
| **Krea2 + krea2_identity_edit** | **0.7 – 0.8** | 手动重建 `[text｜source(frame=1)｜target(frame=0)]` in-context 前向 | ★★★★ | 高降噪下一致性最好，但有时会出现脏点 **不建议拉满**；0.7–0.8 是清晰度与稳定的最佳平衡点 |
| **FLUX.3/FLUX.2 klein** | **0.7 – 0.8** | 仅 reference-latent 通道（text encoder 为纯文本，无 vision grounding） | ★★★ | 缺少 vision 通道，参考信号弱一档，容易脏、偏色, 可使用[Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 做色彩或变形校正，搭配防偏移类lora使用时可提升至★★★★ |
| **Qwen-Image-2512/SDXL系列** | **0.2 – 0.5** | 无 | ★ | 非编辑模型无一致性锚定，即使接入ipadapter等方法，效果依然有限 |

以上测试数量有限，存在片面和主观因素，还是要根据不同情况来调整参数。


### Krea2 identity-edit 的搭配要点

- Krea2 turbo + in-context 时，`reference_strength` 建议从“末段放松”曲线开始（如 4 步，填`1, 1, 0.7, 0.5`）。turbo 步数少，每步强度对结果影响都很大，避免整条曲线大幅低于 1.0（会漂移、发糊）；出现漂移就回到 1.0。旧版“<1.0 会重影、请换 raw 模型”的结论基于已移除的低通/缩小机制，作废。
- **`negative_prompt` 建议留空**：identity-edit 训练时的无条件分布就是"同一张图 + 空文本"，留空即与训练分布对齐；cfg=1.0 时节点也不会浪费编码时间。
---

## 📦 安装

### 方式一：手动安装（Manual Installation）
      cd ./ComfyUI/custom_nodes
      git clone https://github.com/supElement/ComfyUI-Element_TileUpscale.git

### 方式二：通过 Manager 安装（Install using Manager）

在 ComfyUI Manager 中搜索 `ComfyUI-Element_TileUpscale`，点击 Install。

依赖即 ComfyUI 本体，无额外安装项。

---

## 参数说明

### Element Tiled Upscale（一体化节点）

#### 基本

| 参数 | 说明 |
|---|---|
| **model / vae / image / clip** | 基础四件套。clip 必接：逐块 conditioning 由它编码，positive / negative 输入端口已移除 |
| **upscale_by** | 放大倍率，1.0 = 只做重采样精修不放大 |
| **tile_size** | 每块的重采样分辨率。0 = 自动（长边约 4 块，1024–1536）。块越大总块数越少、单块耗时越长。identity-edit LoRA 下每块请 ≤ 2MP |
| **overlap** | 相邻切片间的重叠量。0 = 自动：切片尺寸的 15%，上限为 256px（取两者中的较小值）。任何非零值将按原样使用，不设上限。数值越高，接缝越平滑，但计算开销越大（开销随重叠量呈二次方增长） |
| **feather** | 块间融合的羽化强度，1.0 = 最大平滑，一般不动 |
| **steps / cfg / sampler / scheduler / denoise / seed** | 与 KSampler 同义，直接套用该模型的官方推荐值 |
| **prompt** | 必填。逐块与该块像素（vision 输入）一起重新编码，是唯一的正向条件来源，留空会报错 |
| **negative_prompt** | 留空 = 编码空字符串作为基线负向条件。Krea2 identity-edit 建议保持留空 |

#### 可选（参考图模式组）

| 参数 | 说明 |
|---|---|
| **reference_mode** | 开 = Krea2 identity-edit 专用 in-context 路径（节点内部自建参考传递机制，无需接 Patch 节点；即使接了也会被本节点自动摘除）。关 = 通用 conditioning 路径（Krea2 下必须配合 Patch 节点，该 Patch 会正常生效，并与逐步参考强度调度正常叠加） |
| **redo_first_tile** | 在所有其他图块处理完成后，再次重新生成第一个图块，以便它能将已完成的相邻图块作为上下文参考。此操作仅在存在多个图块时适用。对图像编辑模型影响很小 |
| **grounding_px** | 逐块文本编码的 vision 输入尺寸，默认 768，含义随模型不同：Edit 2509/2511 为 area 桶，0 或 ≥384 一律使用官方 384×384；Qwen 2.1 忽略此参数（锁定为分块自身尺寸）；Krea2 为视觉输入最长边上限，0 = 分块自身尺寸 |
| **reference_strength** | 每步参考强度（STRING）。直接填数值列表：1.0, 0.8, 0.5（中英文逗号/顿号/冒号均可），也可接收 [Element_listGraph_Curve](https://github.com/supElement/ComfyUI_Element_easy) 的 strength_string 输出。第 i 个值作用于第 i 个采样步（0 起）；少于 steps 重复末值，多于截取；单值 = 恒定。 1.0 = 完全参考；0 = 该步完全无参考；纯 0/1 列表单倍算力，含小数时值≠1 的步双前向。参数设置越低，越会出现漂移，为保证一致性，前几步数值设置要高，建议前几步设置为1，后面几步可根据所使用的模型适当降低，这种设置对修复瑕疵更有利。 denoise 不影响本参数；vision/grounding 类参考不受本参数影响。对非图像编辑模型，如sdxl、Qwen image 2512等模型无效 |

#### 输出

| 端口 | 内容 |
|---|---|
| **image** | 最终拼合完成的整图（主输出） |
| **split_images** | 切块后的源图列表（调试用） |
| **resampler_images** | 每块的生成结果列表（调试用，可逐块检查质量） |
| **grid** | 分块网格描述，供 Element Tile Merge 等下游节点复用 |

---

### Element Tile Split

只做放大 + 切块 + 逐块编码，采样交给 ComfyUI 原生 KSampler，适合需要自定义采样细节或插入中间处理的场景。

| 参数 | 说明 |
|---|---|
| **model / image / vae / clip** | clip 必接，positive / negative 输入端口已移除，逐块 conditioning 由 clip + prompt / negative_prompt 编码 |
| **upscale_by** | 同上 |
| **tile_size / overlap** | 同上（tooltip 中含 ≤2MP 提示） |
| **reference_mode** | 开 = Krea 2 identity-edit：逐块输出 vision 槽 grounding + 参考 latent 双通道条件，参考同时挂正负两侧；**必须配合 Patch 节点**（Split 自身不重建前向）。关 = 通用 conditioning 路径 |
| **grounding_px / prompt / negative_prompt** | 含义与一体化节点相同。Split 没有 reference_strength 参数：逐步强度调度请用 **Element Strength Schedule** 节点挂在 model 上（其输出接 KSampler 的 model 输入） |

**输出**：`tile_positives` / `tile_negatives`（逐块 CONDITIONING 列表）、`tile_latents`（逐块 LATENT 列表）、`grid`（分块网格）、`tile_sources`（原图分块 IMAGE 列表，供 Merge 做色彩匹配参考）。

---

### Element Tile Merge

把逐块**解码后**的 IMAGE 列表拼回整图。融合逻辑与一体化节点完全一致：低频走平滑羽化权重、高频走近乎独占的权重。

| 参数 | 说明 |
|---|---|
| **tile_images** | 解码后的逐块 IMAGE 列表 |
| **grid** | Split 输出的分块网格 |
| **feather** | 块间融合的羽化强度 |
| **tile_sources** | 可选。Split 输出的原图分块，作为色彩匹配参考 |
| **color_match_strength** | 0 = 关闭。>0：每个重采样分块与原图对应分块做逐通道 mean/std 色彩迁移（对齐色调、保留细节），需连接 tile_sources |

**输出**：拼合完成的整图。

可以使用另一个扩展中的 [Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 替代，Smart merge images 对色彩和畸变校正的处理结果更好。

---

## 常见问题

**显存不够？**
减小 tile_size（分块更多、单块更小）。Edit 2509/2511 可额外降低 grounding_px；Qwen 2.1 的 grounding_px 不参与计算，只能减小 tile_size。

**SDXL 等普通模型能用吗？**
能。节点会自动退化为普通分块 img2img，参考 latent 被模型忽略，用 denoise 控制细节强度（0.2 – 0.5 起试）。

**Krea 2 用单节点时要不要接 Patch 节点？**
`reference_mode` 开着就不需要（内部自建同款机制，接了也会被自动忽略）；关着（conditioning 路径）**必须接**，否则参考被 core 静默丢弃。详见上文 Patch 节点说明。


