# 节点参数详细说明 (Parameter Description)

本文档详细说明了 ComfyUI-Element_TileUpscale 扩展中各节点的参数作用。

---

## 1. Element Tiled Upscale（一体化节点）

### 基本参数
| 参数 | 说明 |
|---|---|
| **model / vae / image / clip** | 基础四件套。clip 必接：逐块 conditioning 由它编码，positive / negative 输入端口已移除。 |
| **upscale_by** | 放大倍率，1.0 = 只做重采样精修不放大。 |
| **tile_mode** | 开 = 按列数/行数切块 (by count)；关 = 按分块目标尺寸切块 (by size)。 |
| **columns / rows** | 当 tile_mode 开启时，横向和纵向切分的块数。 |
| **tile_size** | 当 tile_mode 关闭时，每块的重采样分辨率。0 = 自动（长边约 4 块，1024–1536）。块越大总块数越少、单块耗时越长。 |
| **overlap** | 相邻切片间的重叠量。0 = 自动：切片尺寸的 15%，上限为 256px。数值越高，接缝越平滑，但计算开销越大。 |
| **feather** | 块间融合的羽化强度，1.0 = 最大平滑。 |
| **steps / cfg / sampler / scheduler / denoise / seed** | 与 KSampler 同义，直接套用该模型的官方推荐值。 |
| **prompt** | 必填。逐块与该块像素（vision 输入）一起重新编码，是唯一的正向条件来源，留空会报错。 |
| **negative_prompt** | 留空 = 编码空字符串作为基线负向条件。Krea2 identity-edit 建议保持留空。 |

### 可选（参考图模式组）
| 参数 | 说明 |
|---|---|
| **reference_mode** | 开 = Krea2 identity-edit 专用 in-context 路径（节点内部自建参考传递机制，**无需再接 Patch 节点**）；关 = 通用 conditioning 路径。 |
| **redo_first_tile** | 在所有其他图块处理完成后，再次重新生成第一个图块，以便它能将已完成的相邻图块作为上下文参考。对图像编辑模型影响很小。 |
| **grounding_px** | 逐块文本编码的 vision 输入尺寸。Edit 2509/2511 为 area 桶（0 或 ≥384 使用 384×384）；Qwen 2.1 忽略此参数；Krea2 为视觉输入最长边上限。 |
| **reference_strength** | 每图块参考强度，决定如何锚定原始图像。1.0 = 完整参考。支持输入逗号分隔的列表，如 `1.0,0.95,0.9` 来逐步衰减参考。 |

---

## 2. Element Tile Split

纯几何分块节点。包含自定义的可视化 UI，支持在画布上手动分块（ctrl + 鼠标左键）、调整参数和预览。

### 顶部工具栏参数
| 参数 | 说明 |
|---|---|
| **▶ / 📂** | 独立运行到本节点 / 从本地加载图像。 |
| **by Manual / by count / by size** | 手动框选模式 / 按数量统一分块 / 按尺寸统一分块。 |
| **Div** | 对齐基数。所有框的坐标、尺寸及最终生成尺寸都将是该值的整数倍（通常设为 16 或 32）。 |
| **U-upscale** | 整体画布的统一放大倍数。 |
| **C / R** | 在 by count 模式下生效。设定全图的横向列数与纵向行数。 |
| **tile** | 在 by size 模式下生效。设定生成块的目标尺寸（像素）。 |
| **overlap** | 块与块之间的重叠像素 (px) 或百分比 (%)。 |
| **fill gaps** | 手动模式下，自动拆分未被手绘框覆盖的区域。 |
| **max side** | 补块的允许最大边长（像素）。超过此值将被切分。 |

### Layers 面板参数
| 参数 | 说明 |
|---|---|
| **ind. up (开关)** | 开启时，下方各手绘图层单独的 `upscale` 参数才生效。关闭时，所有图层强制遵从顶部的 U-upscale。 |
| **upscale** | 该手绘图层的独立放大倍数（需开启 ind. up 开关）。 |
| **cols / rows** | 将该手绘区域进一步横向/纵向切分的数量。 |
| **✕** | 删除该图层。 |

### Prompt 面板参数
| 参数 | 说明 |
|---|---|
| **layer prompt / global prompt** | 切换编辑当前图层的提示词或全局提示词。 |
| **add (开关)** | 开 = 将全局提示词作为后缀追加到每个图层的提示词后。关 = 全局提示词仅作为未填写提示词的图层/背景的默认值。 |
| **positive / negative prompt** | 正向/负向提示词输入框。 |

**输出**：`split_images`（切好的图块列表）、`original_image`（原图）、`prompt` / `negative_prompt`（对应每个图块的提示词列表）、`grid`（网格信息供 Merge 使用）。

---

## 3. Element Tile Merge

把逐块解码后的 IMAGE 列表拼回整图。采用平滑的 Alpha 羽化混合与边缘加权色彩匹配。

| 参数 | 说明 |
|---|---|
| **tile_images** | 解码后的逐块 IMAGE 列表。 |
| **grid** | Split 输出的分块网格。 |
| **feather** | 块间融合的羽化强度。1.0 为完全平滑过渡。 |
| **original_image** | 可选。原图打底。如果不连，边缘羽化处可能会露出黑底。 |
| **tile_sources** | 可选。连接 Split 的 `split_images` 输出，用于对比计算色彩匹配。 |
| **color_match_strength** | 0 = 关闭。>0：开启边缘加权色彩匹配。仅提取块边缘区域（避开中心改色区）计算色彩差异，能有效纠正全局偏色且保护中心物体颜色。 |

**输出**：`Merged_Image`（拼合完成的整图）。

---

## 4. Element Krea2 In-Context Patch (beta) & Element Strength Schedule

- **Element Krea2 In-Context Patch**：非 Krea2 模型或使用单节点开启 `reference_mode` 时不需要接。用于手动重建 Krea2 的前向传播，消费参考图。
- **Element Strength Schedule**：在拆分版工作流（Split -> Encode -> KSampler -> Merge）中，由于 Split 节点不输出模型，需要通过此节点将逐步参考强度调度（reference_strength）挂载到 MODEL 上，再接入 KSampler。
