# Element Tile Upscale 参数手册

---

## TiledUpscale（一体化节点）

一次性完成：放大 → 分块 → 逐块重画 → 羽化合并。

### 输入端口

| 端口 | 类型 | 必接 | 说明 |
|---|---|---|---|
| `model` | MODEL | 是 | 采样模型。可先接补丁节点。 |
| `vae` | VAE | 是 | 编解码用。 |
| `image` | IMAGE | 是 | 源图像。 |
| `clip` | CLIP | 是 | 正/负条件都从这里编码。不接会直接报错。 |

### 输出端口

| 端口 | 类型 | 说明 |
|---|---|---|
| `image` | IMAGE | 最终合并图。 |
| `split_images` | IMAGE 列表 | 每块的源 crop（重采样前）。用于调试或对比。 |
| `resampler_images` | IMAGE 列表 | 每块重画后的结果。 |
| `grid` | GRID | 结构化 JSON。一体化内部已用完，一般不需要接。 |

### 参数

| 参数 | 类型 | 默认 | 说明 | 注意事项 |
|---|---|---|---|---|
| `upscale_by` | FLOAT 1.0–16.0 | 1.0 | 先放大几倍再分块。 | `1.0` 不放大，但依然会分块重画。>1 会先用 lanczos 放大。 |
| `tile_mode` | BOOLEAN | True | ON = 按行列切（用 `columns/rows`）；OFF = 按尺寸切（用 `tile_size`）。 | 切换会**隐藏/显示**对应控件，但隐藏的数值不会丢失（`converted-widget` 保留原值）。 |
| `columns` | INT 1–16 | 2 | 横向列数。 | 只在 `tile_mode = ON` 时生效。 |
| `rows` | INT 1–16 | 2 | 纵向行数。 | 只在 `tile_mode = ON` 时生效。 |
| `tile_size` | INT 0–4096 (step 16) | 0 | 每块的生成分辨率。 | 只在 `tile_mode = OFF` 时生效。`0` = 自动（长边约 4 块，落在 1024–1536）。**identity-edit LoRA 建议每块 ≤ 2MP**。 |
| `overlap` | INT 0–2048 (step 16) | 0 | 相邻块重叠像素。 | `0` = 自动（15% tile 尺寸，上限 256px）。**非 0 值原样使用，不设上限**。接缝明显就加大。 |
| `feather` | FLOAT 0–1 | 1.0 | 合并时的羽化强度。 | 越小接缝越硬。用 smoothstep 曲线过渡，不是线性。 |
| `seed` | INT | 0 | 采样种子。 | 每块的种子会基于此做确定性偏移，改它会影响所有块。 |
| `steps` | INT 1–10000 | 8 | 每块采样步数。 | `reference_strength` 列表与它**严格一一对应**。 |
| `cfg` | FLOAT | 1.0 | CFG。 | **> 1 才会在 in-context / grounded 分支额外编码负条件**。用 Krea 2 in-context 时通常保持 1.0。 |
| `sampler_name` | SAMPLERS | euler | 采样器。 | — |
| `scheduler` | SCHEDULERS | simple | 调度器。 | — |
| `denoise` | FLOAT 0–1 | 1.0 | 重画强度。 | 对内部调度透明：denoise 只改 sigma 范围，不改前向次数，所以不影响 `reference_strength` 的逐步对应关系。 |
| `reference_mode` | BOOLEAN | False | ON = Krea 2 identity-edit（`[text | source | target]` in-context 布局）；OFF = 通用 conditioning 路径。 | **只对 Krea 2 有效**。非 Krea 2 模型打开会**自动回落**到 conditioning 并打 warning。打开时 `clip` 必须接。 |
| `redo_first_tile` | BOOLEAN | True | 所有块跑完后，重画第一块一次，让它参考已完成邻块。 | 只有块数 > 1 时生效。会多一次采样开销。 |
| `grounding_px` | INT 0–4096 (step 32) | 768 | 文本编码器视觉槽尺寸。 | Qwen Edit 2509/2511：面积桶，`0` 或 `≥384` = 官方 384×384；Qwen Image 2.1：**忽略**；Krea 2：限制 vision 输入最长边，`0` = 用 tile 尺寸。 |
| `reference_strength` | STRING | `"1.0,1.0,0.95,0.9,0.8"` | 逐 step 参考强度列表。 | 数量少则重复末值，多则只用前段。分隔符：`, ， 、 : ： ; ； 空格`。`1.0` = 原行为；`0.0` = 该步完全无参考；`0/1` 走 binary 单前向；`0~1` 之间走 guidance **双前向**（≈cfg=2 算力）。全 1 列表零开销。 |
| `prompt` | STRING 多行 | `"enhance quality preserving all the details"` | 正条件。 | **不能为空**（正条件端口已移除，这是唯一来源）。 |
| `negative_prompt` | STRING 多行 | `""` | 负条件。 | 留空 = 编码空字符串。 |

### 分支行为（自动，无需手动选）

| 模型条件 | 走哪条 | 备注 |
|---|---|---|
| `reference_mode=True` 且是 Krea 2 | in_context | `[text | source | target]` 布局。 |
| prompt 非空 且 tokenizer 处理图像 且 **非** FLUX.2 系 | grounded_conditioning | 每块原始像素进 vision 槽，不挂 `reference_latents`。 |
| 其余 | 通用 reference-latents | 每块挂本块 latent。 |

> **FLUX.2 / klein 永远走 reference-latent 路径**，强行 grounded 会破坏条件分布。

---

## ElementTileSplit（拆分版第一步）

纯几何分块 + 图像源接入，无模型依赖。后端执行时会**全量重算几何**，前端 UI 只做预览。

### 输入端口

| 端口 | 类型 | 必接 | 说明 |
|---|---|---|---|
| `image` | IMAGE | 否 | **优先于 UI 上传图**。不接时用 UI 拖拽/粘贴的图。 |
| `layout_data` | STRING | — | 内部隐藏 widget，承载 UI 状态 JSON。**不要手动编辑**。 |

### 输出端口

| 端口 | 类型 | 说明 |
|---|---|---|
| `split_images` | IMAGE 列表 | **已重采样到 `gen_w × gen_h`** 的每块图（不是原始 crop）。接 Encode 用。 |
| `original_image` | IMAGE | 源图。接 Merge 的 `original_image` 用。 |
| `prompt` | STRING 列表 | 每块正条件（layer + global 合成后）。 |
| `negative_prompt` | STRING 列表 | 每块负条件。 |
| `grid` | GRID | 结构化 JSON。接 Encode / Merge。 |

### 三种模式

| 模式 | 行为 | 备注 |
|---|---|---|
| `by Manual` | 手画框。每层可调 upscale / cols / rows / prompt。 | 支持 `fill gaps` 自动补框外区域。 |
| `by count` | 按 `unified.columns × unified.rows` 均匀切。 | **忽略所有手画框**。 |
| `by size` | 按 `unified.tile_size` 摊分块。 | **忽略所有手画框**。 |

### UI 顶部控件

| 控件 | 默认 | 说明 | 注意事项 |
|---|---|---|---|
| `▶` | — | 仅运行到此节点（isolated prompt）。 | 不跑下游，只为看预览。 |
| `📂` | — | 选本地图上传。 | — |
| `by Manual / by count / by size` | Manual | 切模式。 | 切换不会丢手画框数据，但切到 count/size 时**不生效**。 |
| `Div` | 16 | 对齐基数（2–256，step 2）。 | 一般不动。所有坐标/尺寸都吸附到它。 |
| `U-upscale` | 2.0 | 统一放大倍数（0.1–16）。 | 输出画布 = 源 × 它，ceil 对齐 `div`。 |
| `Cols / Rows` | 2 / 2 | count 模式的行列。 | 只在 count 模式生效。 |
| `tile` | 1024 | size 模式的目标尺寸。 | 只在 size 模式生效。 |
| `overlap px / %` | 64px | 重叠。点 `px` / `%` 按钮切换单位。 | 两种都改，只有当前激活的生效。 |
| `fill gaps` | 开 | 自动补框外未覆盖区域。 | 只对 manual 模式有效。 |
| `max side` | 1536 | 补块的最大边长。 | 只对 fill gaps 补块有效。 |

### UI 右侧 Layers 面板

| 控件 | 说明 | 注意事项 |
|---|---|---|
| `alone` | 允许每层使用独立 upscale。 | 不勾 = 所有层统一用 `unified.upscale`。勾了才能改每行的 `upscale`。 |
| 每行 `upscale` | 该层独立放大倍数。 | 只有 `alone` 勾了才生效。 |
| 每行 `cols / rows` | 该层内部再切几列几行。 | 1×1 = 不切。切出来的每块都有 fade。 |
| `✕` | 删除该层。 | — |
| 拖拽行 | 调整图层顺序。 | 影响最终 tile 顺序，不影响几何。 |
| `G` 行 | 显示补块数量与 max side。 | 只读。 |
| `U` 行 | count / size 模式下的只读摘要。 | 显示 tile 数，不显示具体位置。 |

### UI 底部提示词面板

| 控件 | 说明 |
|---|---|
| `layer prompt` | 编辑**当前选中框**的提示词。 |
| `global prompt` | 编辑**所有块共用**的提示词。 |
| `add` | 勾选 = 全局词追加到局部词后（`局部, 全局`）；不勾 = 有局部词用局部词，没局部词才用全局词。 |
| `positive prompt` | 当前编辑目标的正条件。 |
| `negative prompt` | 当前编辑目标的负条件。 |

### UI 画布交互

| 操作 | 效果 |
|---|---|
| `Ctrl + 拖拽` | 新建图层框（吸附 div）。 |
| 拖框内部 | 移动。 |
| 拖框的 8 个控制点 | 缩放（8px 命中容差）。 |
| 点击空白 | 取消选中。 |
| `Delete / Backspace` | 删除当前选中图层。 |
| 拖图 / Ctrl+V | 上传并替换源图。 |

### UI 状态持久化

- 状态同时写入 `layout_data` widget 和 `localStorage`（带**工作流指纹**）。
- 同一个工作流重开会自动恢复。不同工作流之间不会串。
- 执行时后端**全量重算**，前端状态陈旧不会污染结果。

### 提示词合成规则

- `global_append = True`（`add` 勾上）：`layer, global`
- `global_append = False`（`add` 不勾）：`layer if layer.strip() else global`
- **无框层（gap / fallback）** 只输出 global。

### 输出注意事项

- `split_images` 已经按 `gen_w × gen_h` 扩展了边缘。
- 分块总数上限 `MAX_TILES = 256`，超过直接抛错。
- 后端发现 tiles 为空时会退化为"整幅单块"，并打 warning。

---

## ElementTileEncode（拆分版第二步）

把每块图像 / 提示词编码成 per-tile 的 LATENT + CONDITIONING。

### 输入端口

| 端口 | 类型 | 必接 | 说明 |
|---|---|---|---|
| `model` | MODEL | 是 | 用于取 alignment、判断是否 Krea 2、判断是否 grounded。 |
| `vae` | VAE | 是 | 编码每块图像。 |
| `clip` | CLIP | 是 | 编码正/负条件。 |
| `tile_images` | IMAGE 列表 | 是 | 来自 Split 的 `split_images`。 |
| `tile_prompts` | STRING 列表 | 是 | 来自 Split 的 `prompt`。 |
| `tile_negatives` | STRING 列表 | 是 | 来自 Split 的 `negative_prompt`。 |

### 输出端口

| 端口 | 类型 | 说明 |
|---|---|---|
| `tile_positives` | CONDITIONING 列表 | 与 tile 一一对应。接 KSampler 的 positive。 |
| `tile_negatives` | CONDITIONING 列表 | 与 tile 一一对应。接 KSampler 的 negative。 |
| `tile_latents` | LATENT 列表 | 与 tile 一一对应。接 KSampler 的 latent_image。 |

### 参数

| 参数 | 类型 | 默认 | 说明 | 注意事项 |
|---|---|---|---|---|
| `reference_mode` | BOOLEAN | False | ON = Krea 2 identity-edit（in-context 布局）；OFF = conditioning（视 tokenizer 能力自动 grounded）。 | **只对 Krea 2 有效**。ON 时非 Krea 2 会 fallback 到 conditioning 并打 warning。ON 时**必须**在 KSampler 前接 `Krea2 In-Context Patch`。 |
| `grounding_px` | INT 0–4096 (step 32) | 768 | 视觉槽尺寸。 | 语义同 TiledUpscale。 |

### 分支行为（自动）

| 条件 | 分支 |
|---|---|
| `reference_mode=True` 且是 Krea 2 | in_context（挂 `reference_latents + index_timestep_zero`） |
| `reference_mode=False` 且 prompt 非空 且 tokenizer 处理图像 且 **非** FLUX.2 | grounded |
| 其余 | 通用 reference-latents |

### 配套补丁

**本节点不输出 MODEL**。要让参考强度逐步生效：

| 走哪条分支 | 需要哪个补丁 |
|---|---|
| 通用 reference-latents | `ElementStrengthSchedule`（接 KSampler 的 model） |
| Krea 2 in-context | `ElementKrea2InContextPatch`（可选再叠 `StrengthSchedule`） |

---

## ElementTileMerge（拆分版最后一步）

按 `grid` 把每块生成结果贴回画布。

### 输入端口

| 端口 | 类型 | 必接 | 说明 |
|---|---|---|---|
| `tile_images` | IMAGE 列表 | 是 | 每块生成结果（KSampler → VAE Decode 后的）。 |
| `grid` | GRID | 是 | 来自 Split。 |
| `feather` | FLOAT 0–1 | 是 | 羽化强度。默认 1.0。 |
| `tile_sources` | IMAGE 列表 | 否 | 每块原图。颜色匹配的参考。 |
| `original_image` | IMAGE | 否（**建议接**） | 底图。不接时边缘 alpha < 1 可能发黑。 |
| `color_match_strength` | FLOAT 0–1 | 否 | 颜色匹配强度。`0` = 关。 |

### 输出端口

| 端口 | 类型 | 说明 |
|---|---|---|
| `IMAGE` | IMAGE | 合并结果。 |

### 参数

| 参数 | 类型 | 默认 | 说明 | 注意事项 |
|---|---|---|---|---|
| `feather` | FLOAT 0–1 | 1.0 | 羽化强度。 | 用 smoothstep 曲线，不是线性。`0` 基本无羽化，接缝会硬。 |
| `color_match_strength` | FLOAT 0–1 | 0.0 | 颜色匹配强度。 | **只在 > 0 时**触发，且**必须同时接 `tile_sources`** 才真正生效。统计用**边缘低权重像素**（< 0.8），避免中心内容被洗色。 |

### 合并算法

1. 底图：`original_image` 缩放到 `work_w × work_h`。
2. 前景层（`kind = layer`）累加到 fg 画布；背景层（`gap / fallback / unified`）累加到 bg 画布。
3. 归一化后合成：`base = bg * bg_alpha + original * (1 - bg_alpha)` → `merged = fg * fg_alpha + base * (1 - fg_alpha)`。
4. 裁到 `img_h × img_w`（超采样区域裁掉）。

### 注意事项

- **`tile_sources` 不接时 `color_match_strength` 完全无效**。
- **`original_image` 当split节点选择了手动分割模式，fill gaps 未激活时，不接这个端口，缺失分块的未覆盖区会透出黑底。
- `feather = 0` 等于硬拼。  

可以使用另一个扩展中的 [Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 替代，Smart merge images 对色彩和畸变校正的处理结果更好。

---

## ElementKrea2InContextPatch（模型补丁）

**只对 Krea 2 生效**。非 Krea 2 模型 pass-through（原样返回）并打 info 日志。

### 输入 / 输出

| 方向 | 端口 | 说明 |
|---|---|---|
| in | `MODEL` | 采样模型。 |
| out | `MODEL` | 已打 patch。接 KSampler 的 model。 |

### 作用

1. `extra_conds` override：把 conditioning 里的 `reference_latents` 转成 `ref_latents` 前向 kwarg。不处理核心会丢弃这个键，参考到不了 DiT。
2. `DIFFUSION_MODEL` wrapper：用 `[text | source(frame=1) | target(frame=0)]` in-context 布局重建序列。

### 使用场景

| 场景 | 是否需要 |
|---|---|
| 一体化 + Krea 2 + `reference_mode = ON` | **不需要**（内部已处理） |
| 拆分版 + Krea 2 + Encode 的 `reference_mode = ON` | **必须** |
| 非 Krea 2 模型 | 接了也无害（pass-through） |

---

## ElementStrengthSchedule（模型补丁）

给模型挂逐 step 的 `reference_strength` 调度。

### 输入 / 输出

| 方向 | 端口 | 说明 |
|---|---|---|
| in | `model` | 采样模型。 |
| in | `reference_strength` | STRING，逐 step 强度列表。 |
| out | `MODEL` | 已打 patch。接 **KSampler 的 model 输入**。 |

### 参数

| 参数 | 类型 | 默认 | 说明 | 注意事项 |
|---|---|---|---|---|
| `reference_strength` | STRING | `"1.0,1.0,1.0,0.8,0.5"` | 逐 step 参考强度列表。 | 语义同 TiledUpscale。见下方详解。 |

### `reference_strength` 详解

- 与 `steps` **严格一一对应**：第 `i` 步（0 起）取第 `i` 个值。
- 数量不足 → 重复末值；数量超出 → 只用前段。
- 分隔符：英文/中文逗号 `,，`、顿号 `、`、冒号 `:：`、分号 `;；`、空格都行。
- 单值 = 全部相同。

### 三种工作模式（自动判定）

| 列表内容 | 模式 | 算力 |
|---|---|---|
| 全部 `>= 0.999` | 原样返回（不挂 patch） | 零开销 |
| 全部 `0` 或 `1` | binary：该步 `s=0` 直接摘参考 | 单倍 |
| 含小数 | guidance：`pred = pred_无 + s * (pred_有 - pred_无)` | **双倍**（≈cfg=2） |

### 注意事项

- **必须接在 KSampler 的 model 输入上**。Split 不输出 MODEL，接在 Split 后面完全无效。
- **不要和 TiledUpscale 一起用**：一体化内部已挂同名 patch，重复接会被 `_strip_owned_patches` 摘掉再重挂，属于多余操作。
- 分隔符里"小数请用英文句点 `0.8`"，用中文句点会解析失败。
- 可接收 [Element_listGraph_Curve](https://github.com/supElement/ComfyUI_Element_easy) 的 strength_string 输出。
- 参数设置越低，越会出现漂移，为保证一致性，前几步数值设置要高，建议前几步设置为1，后面几步可根据所使用的模型适当降低，这种设置对修复瑕疵更有利。 denoise 不影响本参数；vision/grounding 类参考不受本参数影响。对非图像编辑模型，如sdxl、Qwen image 2512等模型无效。
