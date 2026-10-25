<div align="center">

[![中文](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI-Element_TileEdit
  <sub> 原名：ComfyUI-Element_TileUpscale </sub>

专为图像编辑模型设计的分块精修 / 编辑 / 放大 ComfyUI 扩展（仅限单图），提供强大的分块控制与局部精修能力。
支持模型：Qwen-Image 2.1 / Edit 2509 / 2511、FLUX.3 / FLUX.2 klein、Krea2 + identity_edit lora、Moss + Moss-Image-Edit lora。

✨ 分块放大：告别传统 img2img 的跨块撕裂。
用分块参考保证跨块一致性。你可以放心将 Denoise 重绘强度拉高至 0.7–1.0，同时完美保持原图的人物身份、结构与纹理细节。支持自定义每采样步的参考强度，精准平衡“修复强度”与“一致性”，这是传统 img2img 分块放大无法做到的。

✨ 分块控制与局部精修：用空间分块替代 Mask。
手动分割模式（by Manual）提供了真正的“分块级语义隔离”与像素级控制力。例如：你可以精准框选“白衬衫”所在的图层，并在对应 Layer 的 Prompt 中仅输入“改成蓝衬衫”。未被框选的背景、头发、桌子等会自动划入 Gap fill（补块） 区域，不接收该指令（或仅接收基础的全局画质增强指令），从而完美保持原状。
> 操作方式：CTRL + 左键拖动 添加分块。
> 独立放大：每个手动分割的图层均可单独设置放大尺寸，让核心编辑区域获得更高的分辨率与细节表现。
> ⚠️ 使用 Krea2 模型编辑需要搭配 [krea2_identity_edit](https://huggingface.co/conradlocke/krea2-identity-edit/tree/main) lora。
> ⚠️ 使用 Moss 模型编辑需要搭配 [Moss-Image-Edit](https://huggingface.co/chinoll/Moss-Image-Edit/tree/main) lora。
---

<img width="2379" height="1093" alt="image" src="https://github.com/user-attachments/assets/1464d90d-e044-4423-9d27-7fa2a581a66e" />

---

<img width="2290" height="1123" alt="image" src="https://github.com/user-attachments/assets/ebb876d9-3bc6-436e-a87e-938b7dbf350a" />

---
## 为什么需要它
编辑模型对整图输入很挑剔：

> 图太大 —— 多数模型超过 2K、有的超过 1K 就开始打折，甚至无视指令；
> 
> 目标太小 —— 要精修的区域在整图里占比太小时，直接编辑往往无效或效果不佳；
> 
> 指令太多 —— 一张图塞不下太多指令，多条指令互相干扰、互相遗忘。

分块之后：每块回到模型舒适的分辨率；小目标独占一块，还能单独设定放大倍数；每条指令只写进自己的图层提示词，未框选的区域自动划入补块、保持原状——用空间隔离替代指令堆叠。

> 放大：img2img 的死结
> 
> 传统分块放大（TTP、Ultimate SD Upscale 等）走 img2img 路径，"重绘力度"和"特征保留"共用 denoise 这一个通道：低了模糊塑料感，高了身份漂移，降噪值被锁死在 0.1–0.5。

本扩展把原图作为独立参考信号注入模型——不参与加噪、不随 denoise 衰减，并支持按采样步调度参考强度。denoise 只管重绘力度，可以拉到 0.7–1.0：细节由模型生成，身份与结构由每步可见的参考锚定，清晰度和一致性同时拿到；拼接用羽化权重融合 + 可选的逐块色彩匹配消除接缝。  
邻居上下文（仅限一体化节点）：已生成的相邻块会以全强度反馈给下一块作为输入上下文，让模型把它当作"要延续的内容"，而不是靠重叠区模糊混合去掩盖接缝。（对参考锚定模型而言，邻居上下文主要在 denoise < 1 时软化接缝带的微观纹理；结构与身份的一致性由共享的原图参考承担，这也是 redo_first_tile 对编辑模型几乎无效的原因。）

---

### 已知限制与缺点

#### 1. 依赖特定的模型能力，对非图像编辑模型不友好

> 节点的全部价值建立在**模型具备参考潜空间参考能力**之上。用 SDXL、普通 Flux t2i、Qwen-Image-2512 这类无参考能力的模型时，它会退化成普通的 tiled img2img——此时它**并不比 TTP 等工具更好**。

#### 2. 模型识别依赖逆向工程，可能存在 comfyUI 跨版本稳定性

> 对模型的识别依赖**类名子串匹配**（`"flux" in type(dm).__name__.lower()`）、**tokenizer 签名反射**（`inspect.signature`）和 `vars()` 遍历。这些都不是稳定 API。
> 
> Krea2 的 in-context 路径是**手动重建前向传播**，直接调用 `dm.blocks`、`dm.pe_embedder`、`img_slice` 等内部结构；Patch 节点还依赖 core 的 `extra_conds` object-patch 与 DIFFUSION_MODEL wrapper 机制。一旦上游模型实现变动（改名、改签名、改内部布局），节点很可能**静默出错或直接崩溃**。

#### 3. 性能与显存开销更大(在可接受范围内)

> 每个 tile 都要**独立编码参考 latent**；开启 grounding 时，每个 tile 还要**重新走一遍 text encoder 的 vision 编码**。分块越多，这部分开销线性增长。
> 
> Krea2 的 in-context 路径需要跑完整的 DiT 前向，token 序列是 `text + source + target` 三段拼接，注意力成本比原生长。
> 
> tile 的 token 数增长时，注意力成本按**平方**增长，因此自动 tile 尺寸被限制在 1024–1536。identity-edit LoRA 的训练分辨率上限约 2MP，tile 请尽量不超过 2MP（1536²≈2.36MP 已越界，1408²≈1.98MP 贴上限）。

#### 4. 不保证对未适配模型的正确性

---

## 推荐降噪值与搭配

| 模型 | 推荐降噪值 | 参考注入方式 | 一致性 | 说明 |
|---|---|---|---|---|
| **Qwen-Image 2.1** | **1.0** | 参考 latent 精确插在 vision token 位置（image_slots），tile 保持原尺寸不 resize | ★★★★★ | 参考信号最原生，多数情况下可直接拉满。细节完全由模型生成，同时身份 / 结构保持一致 |
| **Qwen-Image Edit 2511** | **1.0** | 384×384 vision bucket + reference latent 双通道 | ★★★★ | 同 2509，推荐直接用 1.0 ，缺少细节 |
| **Qwen-Image Edit 2509** | **1.0** | 384×384 vision bucket + reference latent 双通道 | ★★★★ | 与 2511 同源，参考通道成熟，可拉满，缺少细节 |
| **FLUX.2 klein base** | **0.7 – 1.0** | 仅 reference-latent 通道（text encoder 为纯文本，无 vision grounding） | ★★★★ | 缺少 vision 通道, 面部有些油, 偏色, 可使用[Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 做色彩或变形校正，搭配防偏移类lora使用时可提升至★★★★☆ |
| **Krea2 + krea2_identity_edit** | **0.7 – 0.8** | 手动重建 `[text｜source(frame=1)｜target(frame=0)]` in-context 前向 | ★★★☆ | 高降噪下色彩一致性最好，但有时会出现脏点，0.7–0.8 是清晰度与稳定的最佳平衡点 |
| **MOSS-Image + Moss-Image-Edit** | **0.7 – 1** | 原生 `reference_latents` 通道，采用 `index_timestep_zero_text_start` 布局；ComfyUI 自动预计算 `ref_kv` 加速 | ★★★ | 新版comfyUI 原生支持 Edit LoRA。借助 1:1 尺寸参考注入，细节保留。视觉语义补全、全局语境理解以及先验知识方面较弱。因涉及首步 K/V 预计算，第一步耗时略长。推荐搭配 ComfyUI 最新版本（#17296）。 |
| **FLUX.3 klein turbo** | **0.7 – 0.8** | text encoder 仅文本有效，无 vision grounding | ★★★ | 缺少细节、易变形（也可能是comfyUI实现的问题），优点是不容易脏 |
| **FLUX.2 klein turbo** | **0.7 – 0.8** | 仅 reference-latent 通道（text encoder 为纯文本，无 vision grounding） | ★★★ | 缺少 vision 通道，参考信号弱一档，很容易脏、偏色, 可使用[Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 做色彩或变形校正，搭配防偏移类lora使用时可提升至★★★☆ |
| **Qwen-Image-2512/SDXL系列** | **0.2 – 0.5** | 无 | ★ | 非编辑模型无一致性锚定，即使接入ipadapter等方法，效果依然有限 |

> CFG 取值应根据使用的模型设置。
> 
> 编辑模型 denoise 取值范围 0.6–1.0；非编辑模型 denoise 取值不超过 0.5。
> 
> 模型搭配策略：建议使用 base 或 raw 等非蒸馏模型，搭配 turbo lora（权重约 0.8，根据具体模型自行调整）。这能兼顾基础模型的高质量细节还原与蒸馏模型的生成速度，并能有效规避直接使用纯 turbo 模型放大时容易产生的高频伪影（脏斑）。
> 
> 参考强度曲线（reference_strength）：建议使用“前高后低”的末段放松曲线。例如总步数为 4 步时，可填 1.0, 1.0, 0.9, 0.8。由于 turbo 模型步数极少，每一步的强度对结果影响都很大，建议全程参考强度不要低于 0.7（否则容易导致图像漂移）；若实际使用中出现特征漂移，可将曲线直接拉平至全 1.0。
> 
> Krea2 模型特调(在一体化节点中使用Krea2 identity-edit 模式时，需打开 `reference_mode`)：建议将 negative_prompt 完全留空。这是因为 krea2 的 identity-edit 在训练时的无条件分布即为“同一张图 + 空文本”，留空能与训练分布完美对齐；且在 cfg=1.0 时，节点也能省去负面提示词的编码时间。

#### 以上测试数量有限，存在片面和主观因素，还是要根据不同情况来调整参数。当然，你也可以通过使用增强lora，或使用微调模型来改善结果，但这不在测试范围内。  
---

## 全局提示词（非 by manual 模式）

> 在非手动分块模式下（如一体化节点、by count、by size），本节点会将同一份全局提示词应用于每一个分块的文本编码。由于每个分块仅涵盖原图的一小片局部区域，该区域的具体内容完全由参考图（像素特征）决定，而提示词仅负责指导模型“如何处理”这些内容。
> 
> 因此，全局提示词必须遵循“对所有分块均普遍适用”的核心原则：

### ✅ 应该写
1. 操作指令 —— 告诉模型这次采样的核心任务（对每个分块都成立）：
  - sharpen and refine details, preserve identity and structure, clean upscale  
  - （锐化并精修细节，保持身份与结构，干净地放大）

2. 全局风格 / 光照 / 色调 —— 整张图共有的氛围（对每个分块也都成立）：
  - cool blue ambient lighting, warm orange rim light, cinematic tone, photorealistic  
  - （冷蓝色环境光，暖橙色轮廓光，电影级色调，照片级写实）

### ❌ 不应该写
  - 具体的物体、人物或场景名词（如 woman at a computer, glasses, monitor, coffee cup）。
  - 除非你能确定该提示词描述的物体元素，对画面中的每一个分块都成立。

### 💡 推荐模板
  - sharpen and refine, preserve identity and structure, {全局色调/光照/风格}, {画质词}
---

## 📦 安装

### 方式一：手动安装（Manual Installation）
      cd ./ComfyUI/custom_nodes
      git clone https://github.com/supElement/ComfyUI-Element_TileEdit.git

### 方式二：通过 Manager 安装（Install using Manager）

在 ComfyUI Manager 中搜索 `ComfyUI-Element_TileEdit`，点击 Install。

依赖即 ComfyUI 本体，无额外安装项。

---
## 节点一览

| 节点 | 作用 |
|---|---|
| **Element Tiled Upscale** | 一体化节点：放大 → 切块 → 逐块参考重绘 → 频率分离融合，一键出图 |
| **Element Tile Split** | 纯几何分块 + 图像源接入，无模型依赖。后端执行时会全量重算几何，前端 UI 只做预览。|
| **Element Tile Encode** | 把每块图像 / 提示词编码成 per-tile 的 LATENT + CONDITIONING，采样交给 ComfyUI 原生 KSampler |
| **Element Tile Merge** | 把逐块解码后的 IMAGE 列表用同一套低频/高频分离 + 羽化权重拼回整图，可选逐块色彩匹配 |
| **Element Krea2 In-Context Patch** | 模型补丁：让 Krea2 以 identity-edit 的 in-context 布局消费挂在 conditioning 上的参考 latent |
| **Element Strength Schedule** | 模型补丁：把逐步 reference_strength 调度挂到模型上，供 ElementTileSplit + 原生 KSampler 工作流使用，KSampler 的 model 输入必须接本节点输出。一体化节点内置同款调度且会忽略上游挂载，单节点工作流直接用其自带参数即可。第 i 个值作用于第 i 个采样步（0 起）；少于 steps 重复末值，多于截取；单值 = 恒定。 1.0 = 完全参考；0 = 该步完全无参考；纯 0/1 列表单倍算力，含小数时值≠1 的步双前向。参数设置越低，越会出现漂移，为保证一致性，前几步数值设置要高，建议前几步设置为1，后面几步可根据所使用的模型适当降低，这种设置对修复瑕疵更有利。 对非图像编辑模型，如sdxl、Qwen image 2512等模型无效 |

## 参数详细说明：[中文](md/Description.md) | [English](md/Description_en.md)

---

## 常见问题

**怎么用？**  
- workflow文件夹中已提供相关工作流。  

**怎么手动分割？**

- 必须使用 Element Tile Split 节点中的 by manual 模式，在画布上按住 CTRL键 + 鼠标左键，拖动添加分块，可连续添加。
- 每添加一个分块，右侧面板会增加相对应的图层，每个图层可在下方的图层提示词中单独设置提示词。
- 鼠标拖动图层可以调整图层顺序，merge节点在合并时最上方的图层会最后合并，所以多个手动分块重叠时，保证最上方的图层完全可见。
- 独立四角圆角：选中图层后，拖拽四个角上的青色圆点，可独立调整该角的圆角半径。背景补块会自动贴合圆角外轮廓。
- 可以选择不需要的分块按 delete 删除分块（图层），也可以用鼠标点击对应图层上的删除按钮来删除分块。

**显存不够？**  
- 减小 tile_size（分块更多、单块更小）。Edit 2509/2511 可额外降低 grounding_px；Qwen 2.1 的 grounding_px 不参与计算，只能减小 tile_size。

**SDXL 等普通模型能用吗？**  
- 能。节点会自动退化为普通分块 img2img，参考 latent 被模型忽略，用 denoise 控制细节强度（0.2 – 0.5 起试）。

**Krea 2 用单节点时要不要接 Patch 节点？**  
- `reference_mode` 开着就不需要（内部自建同款机制，接了也会被自动忽略）；关着（conditioning 路径）**必须接**，否则参考被 core 静默丢弃。详见上文 Patch 节点说明。


