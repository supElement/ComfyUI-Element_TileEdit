# Element_TileEdit 参数手册

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
| `feather` | FLOAT 0–1 | 1.0 | 合并时的羽化强度。 | 越小接缝越硬。用 smoothstep 曲线过渡，不是线性。**一体化按分块边缘起算过渡带**，与拆分版 Merge 的"两块锁定带之间"语义不同。 |
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

<img width="1005" height="1145" alt="image" src="https://github.com/user-attachments/assets/9785fe15-b11a-4789-a05e-f09e48b97855" />

---
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
| `tile_masks` | MASK 列表 | 每块的采样 mask。开启 `edge mask` 时非全白；**必须**接到 Encode 的 `tile_masks`（Encode 已设为必填端口）。 |

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
| `overlap px / %` | 64px | 重叠。点 `px` / `%` 按钮切换单位。 | 两种都改，只有当前激活的生效。**不设上限**，填多少用多少。 |
| `edge mask` | 关 | 把 overlap 外圈用 mask 锁住：采样时该带不重绘，只作参考；Merge 端自动剔除。 | 必须把 Split 的 `tile_masks` 接到 Encode（该端口已设为必填，未连会直接报错）。详见下方"锁边"。 |
| `lock` | 90% | 锁边带宽 = `overlap/2 × lock%`（0–100%）。 | 默认 90。100 = 半边 overlap 全部锁住，Merge 剔除会饱和在 `overlap/2`，重叠区几乎没有参与合成的重绘内容，一般不建议常用。 |
| `feather` | 50% | 锁定带**内部**的羽化比例（0–100%，相对锁定带宽度）。 | 外沿恒为 mask=0，向块内渐变到 1，不侵占带外的可重绘区；100% = 渐变正好覆盖整条锁定带。Merge 只剔除锁定带本身。 |
| `fill gaps` | 开 | 自动补框外未覆盖区域。 | 只对 manual 模式有效。 |
| `max side` | 1536 | 补块的最大边长。 | 只对 fill gaps 补块有效。 |

### 锁边（edge mask）

开启后，只对**有邻居的边**锁住外圈一段（画布外沿不锁）：

- **锁定带** = `overlap/2 × lock%`，比例折算到生成分辨率后写入采样 `noise_mask`：这一段不重绘，只作为参考。大幅改风格时能防止内容漂移。
- **`feather`** 是锁定带**内部**的羽化比例（0–100%，相对该边的锁定带宽度）。羽化只向 `mask=0` 一侧腐蚀：外沿始终为 0，越往块内越接近 1，到锁定带内沿正好到 1；**不会侵占锁定带以外的可重绘区**。`0%` = 硬边；`50%` = 外半段实锁、内半段渐变；`100%` = 整条锁定带都是渐变。
- **Merge 只剔除锁定带本身**（`≤ overlap/2`），不再额外加 `feather`，因为羽化已经完全落在带内。剔除是硬切，过渡交给 Merge 自己的 `feather`。
- 每边最多剔到 `overlap/2 − 1px`。`lock=100%` 时正好顶到中线并触发限幅 + warning，留给 Merge 羽化的中段几乎没有，一般不建议。
- **手绘块的倒角同步锁定**：锁定带会沿圆角绕过去。采样轮廓 = 外扩轮廓按锁定带宽整体内缩（圆角**同心**内缩：圆心不变，半径从 `r + overlap/2` 减到 `r + overlap/2 − 锁定带`），所以倒角处的 `mask=0` 是一条与直边等宽的带，而不是把整个倒角切成一大片黑角。没有倒角（半径 0）的框完全走原来的四边带逻辑。与直边带一致：轮廓内侧恒为 1（可重绘），羽化落在轮廓**外侧**的锁定区，从轮廓向外按 `feather` 宽度渐变到 0（不是硬切）。一个角涉及两条边时按较大的锁定带内缩；贴画布边（无邻居）的一侧带宽为 0。
- 例：`overlap=100px, lock=50%` → 锁定带 25px；`feather=100%` → 这 25px 全部是渐变（外沿 0 → 内沿 1）；`feather=50%` → 外 12.5px 实锁、内 12.5px 渐变。
- 链路：Split 的 `tile_masks` → Encode 的 `tile_masks`，**必接**。Encode 已把该端口设为必填，未连接时 ComfyUI 在排队阶段就会报错，不会跑完采样才发现锁边没生效。

---

下面图示以 Tile01 为例，Tile02 镜像，feather 示意只适用于 split 节点；merge 节点的 feather 在同层两块手绘块之间取锁定带之间的可重绘区；对背景一侧则以手绘框边为 50% 点、向外只到本块剔除边界；倒角按以手绘圆弧为 50% 的径向斜坡过渡。  

<img width="2168" height="1133" alt="image" src="https://github.com/user-attachments/assets/3bf328fb-66ef-48bc-ac77-df8c6cdedacb" />

---
### 锁边的模型适用性

锁边不绑定 Qwen Image 2.1，**所有走 ComfyUI 标准采样器的模型都能用**（Qwen Image 2.1、Qwen Edit 2509/2511、Krea 2、FLUX.1、FLUX.2/klein、SD3.5、SDXL 等）：

- 采样端：Split/Encode 只是把锁定带写进 latent 的 `noise_mask`，ComfyUI 会让每个采样步在该带内保留输入 latent，因此与模型结构无关。只有绕过 `noise_mask` 的第三方自定义采样器不生效。
- Merge 端：剔除锁定带、两块锁定带之间的过渡都是纯几何运算，与模型无关。
- **分辨率差异**：锁定带的量化精度 = 模型 latent 的降采样倍数。8× 模型（Qwen Edit 2509/2511、Krea 2、FLUX.1、SD3/SDXL）约 8px 一格，16× 模型（Qwen Image 2.1、FLUX.2/klein）约 16px 一格。同样的 `lock`，16× 模型上锁定带更"软"，建议锁定带至少覆盖 2 个 latent 格（8× 模型 ≥16px，16× 模型 ≥32px）。
- **边界残留宽度因模型而异**：即使做了羽化，mask 边界附近仍可能残留一小条没完全重绘的区域。identity-edit / in-context 类（Krea 2）和大 denoise 配置残留更宽，inpaint 训练过的模型更服帖。所以 Split 的 `feather` 和 Merge 的 `feather` 都做成了可调项。
- 没做过 inpaint/mask 训练的模型（FLUX.1 dev、SD3.5、SDXL base 等）也能工作，只是边界风格过渡可能更生硬，建议加大 `overlap`（并适当降低 `lock`，给 Merge 羽化留出中段）。

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
| `Ctrl + 拖拽` | 新建图层框（吸附 div） |
| 拖框内部 | 移动。 |
| 拖框的 8 个控制点 | 缩放（8px 命中容差） |
| 拖动框内四角的控制点 | 改变/调整圆形倒角大小，四角独立控制。不影响输出的分块图像列表，倒角只在Element Tile Merge 节点处理 |
| 点击空白 | 取消选中。 |
| `Delete / Backspace` | 删除当前选中图层 |
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
- `tile_masks` 与分块一一对应；未开启 `edge mask` 时全是全白 mask。
- 分块总数上限 `MAX_TILES = 256`，超过直接抛错。
- 后端发现 tiles 为空时会退化为"整幅单块"，并打 warning。

---

## ElementTileEncode（拆分版第二步）

把每块图像 / 提示词编码成 per-tile 的 LATENT + CONDITIONING，并同时输出可直接采样的 MODEL（Krea 2 in-context 分支会自动补补丁）。

### 输入端口

| 端口 | 类型 | 必接 | 说明 |
|---|---|---|---|
| `model` | MODEL | 是 | 用于取 alignment、判断是否 Krea 2、判断是否 grounded。 |
| `vae` | VAE | 是 | 编码每块图像。 |
| `clip` | CLIP | 是 | 编码正/负条件。 |
| `tile_images` | IMAGE 列表 | 是 | 来自 Split 的 `split_images`。 |
| `tile_prompts` | STRING 列表 | 是 | 来自 Split 的 `prompt`。 |
| `tile_negatives` | STRING 列表 | 是 | 来自 Split 的 `negative_prompt`。 |
| `tile_masks` | MASK 列表 | **是** | 来自 Split 的第 6 个输出 `tile_masks`。白色 = 重绘，黑色 = 保留作参考。未开 `edge mask` 时是整列全白，接了也无害；**不接则 ComfyUI 直接拒绝执行**，避免开了锁边却白跑采样。 |

### 输出端口

| 端口 | 类型 | 说明 |
|---|---|---|
| `model` | MODEL | 采样模型，接 KSampler 的 model。走 Krea 2 in-context 且上游未挂 in-context 补丁时，本节点已自动补上；其余情况原样返回。 |
| `tile_positives` | CONDITIONING 列表 | 与 tile 一一对应。接 KSampler 的 positive。 |
| `tile_negatives` | CONDITIONING 列表 | 与 tile 一一对应。接 KSampler 的 negative。 |
| `tile_latents` | LATENT 列表 | 与 tile 一一对应。接 KSampler 的 latent_image。 |

### 参数

| 参数 | 类型 | 默认 | 说明 | 注意事项 |
|---|---|---|---|---|
| `reference_mode` | BOOLEAN | False | ON = Krea 2 identity-edit（in-context 布局）；OFF = conditioning（视 tokenizer 能力自动 grounded）。 | **只对 Krea 2 有效**。ON 时非 Krea 2 会 fallback 到 conditioning 并打 warning。ON 时直接把本节点的 `model` 输出接 KSampler 即可（in-context 补丁已自动补）；只有坚持用原始 MODEL 时，才需要在 KSampler 前手动接 `Krea2 In-Context Patch`。 |
| `grounding_px` | INT 0–4096 (step 32) | 768 | 视觉槽尺寸。 | 语义同 TiledUpscale。 |

### 分支行为（自动）

| 条件 | 分支 |
|---|---|
| `reference_mode=True` 且是 Krea 2 | in_context（挂 `reference_latents + index_timestep_zero`） |
| `reference_mode=False` 且 prompt 非空 且 tokenizer 处理图像 且 **非** FLUX.2 | grounded |
| 其余 | 通用 reference-latents |

### 配套补丁

**本节点输出 `model`**。Krea 2 in-context 分支且上游还没挂 in-context 补丁时，本节点会自动补上（等同 `ElementKrea2InContextPatch`）；其它分支原样返回，不会重复打补丁。

| 走哪条分支 | 需要哪个补丁 |
|---|---|
| 通用 reference-latents / grounded | `ElementStrengthSchedule`：接在本节点 `model` 输出与 KSampler 之间，逐 step 削弱参考 |
| Krea 2 in-context | in-context 补丁已由本节点自动补；如需再控参考强度，在 `model` 输出后串 `ElementStrengthSchedule`（可选） |

---

## ElementTileMerge（拆分版最后一步）

按 `grid` 把每块生成结果贴回画布。

### 输入端口

| 端口 | 类型 | 必接 | 说明 |
|---|---|---|---|
| `tile_images` | IMAGE 列表 | 是 | 每块生成结果（KSampler → VAE Decode 后的）。 |
| `grid` | GRID | 是 | 来自 Split。 |
| `feather` | FLOAT 0–1 | 是 | 羽化强度（同层手绘块之间在可重绘中段内取值；对背景一侧按整个重叠区居中）。默认 1.0。 |
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
| `feather` | FLOAT 0–1 | 1.0 | 羽化强度。 | **同一层**的两块手绘块之间：过渡带落在两块剔除带之间的可重绘区，宽度 = 该区 × feather。手绘块对着背景（`gap / fallback / unified`）的一侧：向外止于本块的剔除边界（权重在那里正好归零，锁定带里未重绘的参考像素不会被半透明地带进合成），向内到重叠区中线，50% 始终落在手绘框边上。**倒角**走同样的规则，只是换成径向斜坡：圆心 = 手绘圆弧圆心，50% 落在半径 `r` 的圆弧上，向外到 `r + 可重绘宽度` 归零——倒角由本节点的 `feather` 控制，Split 的倒角羽化只管采样锁边。背景块不剔除锁定带，过渡按整个重叠区铺满。`0` = 在过渡带中线硬切（同一层两块之间是锁定带之间的中线，对背景一侧正好是手绘框边），`0.5` = 只混合中间一半，`1` = 整段可重绘区参与混合。权重互补，不会与底图混合发糊。用 smoothstep 曲线，不是线性。 |
| `color_match_strength` | FLOAT 0–1 | 0.0 | 颜色匹配强度。 | **只在 > 0 时**触发，且**必须同时接 `tile_sources`** 才真正生效。统计用**边缘低权重像素**（< 0.8），避免中心内容被洗色。 |
| `color_protection` | BOOLEAN | False | 色相保护 | 开启：将每个图块与其源内容进行比对，仅在像素被刻意重新着色的区域进行色彩匹配调整；这样既保留了人工绘制的色调/色彩修改，又能确保曝光修正依然生效。关闭：照常对每个图块进行色彩匹配。 |
| `high_low_freq` | BOOLEAN | False | 高低频融合（只作用于羽化区） | 在低频部分采用“羽化”权重以平滑过渡，在高频部分则采用“竞争性”权重以保留细节——从而避免重叠区域内的细节因平均化处理而被抹平。 |

### 合并算法

1. 底图：`original_image` 缩放到 `work_w × work_h`。
2. 手绘块按 `grid` 里的 `edit_mask` 剔除 Split 锁定的区域（只剔除锁定带本身，`≤ overlap/2`）。背景块（`gap / fallback / unified`）不剔除：它们的锁定带里只是同一底图的另一个增强版本，剔除反而会让权重和 < 1、透出底图。
3. 手绘块的羽化过渡带：**同一层**的相邻手绘块之间限制在**两块锁定带之间**（`grid` 里每块的 `edit_bands` + 对侧邻块的 `edit_bands`），过渡宽度 = `overlap` 减去两侧锁定带（`grid["overlap"]` 是唯一的重叠来源：手动模式下每个子块都只外扩 `overlap/2`，同层子块不再另叠一份重叠），50% 落在两块重叠区的正中；对背景的一侧同样从本块剔除边界起算，向外不进入锁定带，50% 落在手绘框边上。锁定带边缘的权重恒为 0，剔除是硬切也不会留下接缝。
4. 倒角按径向斜坡处理：与直边同一条规则，50% 落在手绘圆弧上，向外到 `r + 可重绘宽度` 收敛到 0、向内升到 1，宽度再乘 `feather`；`feather = 0` 就是沿手绘圆弧硬切。外扩轮廓（`r + overlap/2`）不再参与切边，所以倒角不会按外扩圆角或方块边界被切掉。
5. 手绘框离画布边还有外扩留白时（`layer_region` 比 tile 边内缩），该边即使贴着画布也补回过渡：宽度 = 2 × 留白，50% 落在手绘框边上；不留白（框本身就贴边）则不补。
6. 前景层（`kind = layer`）累加到 fg 画布；背景层（`gap / fallback / unified`）累加到 bg 画布。
7. 归一化后合成：`base = bg * bg_alpha + original * (1 - bg_alpha)` → `merged = fg * fg_alpha + base * (1 - fg_alpha)`。
8. 裁到 `img_h × img_w`（超采样区域裁掉）。

### 注意事项

- **`tile_sources` 不接时 `color_match_strength` 完全无效**。
- **`original_image` 当split节点选择了手动分割模式，fill gaps 未激活时，不接这个端口，缺失分块的未覆盖区会透出黑底。
- `feather = 0` 等于在可重绘区中线硬切（不是贴分块边缘硬切）。
- 开启 Split 的 `edge mask` 时，只有**同一层的相邻手绘块之间**会把过渡带收缩到两块锁定带之间；手绘块对着背景的一侧从本块剔除边界起算、以手绘框边为 50% 点，所以 `lock` 再大也不会吃掉手绘框的边和倒角，也不会在剔除边界拉出硬切。同一层内锁定带越宽，中段越窄（单侧过渡宽度 = `overlap/2 − 本块锁定带`，`lock` 太大时同一层的两块之间会只剩一条硬接线），要宽过渡就把 `lock` 放小。背景块不受影响，始终按整个重叠区过渡。
- **倒角只看本节点的 `feather`**：Split 的 `corner feather` 只用于采样锁边（往里渐变），Merge 端不再拿外扩轮廓或方块边界切倒角。

可以使用另一个扩展中的 [Smart merge images节点](https://github.com/supElement/ComfyUI_Element_easy) 替代，Smart merge images 对色彩和畸变校正的处理结果更好。接 Split 的 `grid` 时它直接沿用本节点的 alpha / feather 几何（居中斜坡、同层锁定带之间的过渡、手绘圆弧倒角），手绘块的锁定带一样会被剔除，背景块不剔除。

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
| 拆分版 + Krea 2 + Encode 的 `reference_mode = ON` | **不需要**（用 Encode 的 `model` 输出，已自动补）；只有坚持用原始 MODEL 接 KSampler 时才手动接 |
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

- **必须接在 KSampler 的 model 输入上**。Split 不输出 MODEL，接在 Split 后面完全无效；拆分版请接在 Encode 的 `model` 输出与 KSampler 之间。
- **不要和 TiledUpscale 一起用**：一体化内部已挂同名 patch，重复接会被 `_strip_owned_patches` 摘掉再重挂，属于多余操作。
- 分隔符里"小数请用英文句点 `0.8`"，用中文句点会解析失败。
- 可接收 [Element_listGraph_Curve](https://github.com/supElement/ComfyUI_Element_easy) 的 strength_string 输出。
- 参数设置越低，越会出现漂移，为保证一致性，前几步数值设置要高，建议前几步设置为1，后面几步可根据所使用的模型适当降低，这种设置对修复瑕疵更有利。 denoise 不影响本参数；vision/grounding 类参考不受本参数影响。对非图像编辑模型，如sdxl、Qwen image 2512等模型无效。
