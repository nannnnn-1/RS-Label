# SAM 半自动数据标注工具 设计方案

## 项目目标

构建一个基于 PySide6 的半自动数据标注工具，以 SAM2（点/框提示）和 SAM3（文本提示）作为 AI 辅助引擎，纯手动标注模式作为 fallback，输出与 labelme 完全兼容的 JSON 标注文件。

### 核心目标（按优先级）

1. **手动标注模式**：完整实现 labelme 的核心标注能力（多边形、矩形、标签管理、JSON 导出），确保无模型时也能独立工作
2. **SAM2 交互式标注**：通过鼠标点击正/负点和绘制矩形框，实时调用 SAM2 生成分割 mask，支持迭代修正
3. **SAM3 文本提示标注**：通过输入文本描述（如"黑猫"），让 SAM3 自动寻找并分割图中所有匹配目标，支持多候选浏览、置信度过滤、勾选确认
4. **标注文件互通**：输出的 JSON 文件与 labelme 格式完全兼容，已有的 labelme 标注数据可直接在此工具中打开编辑

## 技术栈

| 层 | 技术 | 说明 |
|---|---|---|
| UI 框架 | PySide6 | Qt for Python，LGPL 许可 |
| 图像处理 | NumPy, OpenCV, Pillow | 图像读写、mask 转 polygon |
| 模型推理 | PyTorch (已有 `data_annotation` 环境) | SAM2/SAM3 模型推理 |
| 模型来源 | 本地第三方仓库 + 本地权重 | `third_party_repository/` + `Models/` |
| 标注格式 | JSON（labelme 兼容） | 完全兼容 labelme 5.x |

## 三大模式详解

### 模式一：纯手动标注（labelme 模式）

**目标**：复刻 labelme 核心标注功能，不需要加载任何模型。

**功能清单**：
- 多边形标注工具（Polygon）：连续点击创建顶点，右键或双击闭合
- 矩形标注工具（Rectangle）：拖拽绘制
- 选择/编辑工具（Select）：选中已有标注，拖动顶点修改
- 删除标注
- 标签管理面板：新建标签、删除标签、标签颜色自动分配
- 标注列表：显示当前图像所有标注，点击可选中/高亮
- 文件列表：浏览目录下所有图片，点击切换
- 保存/另存为：导出为 labelme 兼容 JSON
- 打开已有标注文件：加载 JSON 并恢复标注状态
- 快捷键：Ctrl+S 保存，Delete 删除选中标注，Esc 取消当前绘制

**验收标准**：
- 能用多边形工具完成一次完整标注并保存为 JSON
- 能用矩形工具完成标注并保存
- 保存的 JSON 可被 labelme 正常打开并显示所有标注形状
- 能用此工具打开一个已有的 labelme JSON 文件，标注形状正确显示
- 标签列表可正常增删，颜色分配不冲突
- 文件列表正确显示当前目录下的所有图片文件（.jpg/.png/.bmp/.jpeg）

---

### 模式二：SAM2 交互式标注

**目标**：使用 SAM2 模型通过鼠标点/框提示快速生成分割 mask，迭代修正后确认转为标注。

**使用模型**：SAM2（四种规格可选：tiny / small / base_plus / large）
- tiny (149MB)、small (176MB)、base_plus (309MB)、large (857MB)
- 默认推荐：base_plus（速度快、精度够用）

**交互流程**：
```
1. 切换到模式二（工具栏点击 SAM2 模式按钮）
2. 选择 SAM2 模型规格（下拉菜单）
3. 加载模型（首次使用时加载，之后保持在内存）
4. 打开图片 → 自动调用 set_image() 预计算 image embedding
5. 用户操作：
   - 左键点击 = 正点提示（前景，绿色标记）
   - 右键点击 = 负点提示（背景，红色标记）
   - 拖拽 = 矩形框提示（蓝色线框）
6. 每次操作后实时调用 predict() 更新 mask 预览（半透明遮罩覆盖）
7. 不满意 → 继续加正/负点或框修正
8. 满意 → 按 Enter 或点击工具栏 [确认] 按钮
   - mask 自动转为 polygon（OpenCV findContours + 简化）
   - 弹出标签选择对话框
   - polygon 添加到标注列表
9. 按 Esc → 清空所有当前提示点/框，取消预览
10. 工具栏 [撤销提示] → 撤销最近一次添加的点/框
11. 切换图片时自动 reset_image() 并重新 set_image()
```

**工具栏元素（模式二专属）**：
- 模型规格下拉：tiny | small | base_plus | large
- 正点工具（默认，左键点击）
- 负点工具（右键点击或按住 Ctrl+左键）
- 框工具（拖拽绘制）
- [撤销提示] 按钮
- [确认标注] 按钮
- [取消] 按钮
- 提示点/框在 Canvas 上的标记（绿色+、红色-、蓝色框）

**验收标准**：
- 能成功加载 SAM2 四种规格模型中至少一种（tiny/base_plus 优先）
- 打开图片后，单次左键点击能在 2 秒内显示 mask 预览
- 正点+负点组合使用能明显改善 mask 准确性
- 框提示能正常生成 mask
- mask 转 polygon 后形状合理（轮廓点数适中，不过度简化也不过于复杂）
- 连续标注 3 张图片无内存泄漏或显存溢出
- 确认后的标注能被模式一（手动模式）正常编辑
- 模型切换（如 tiny→base_plus）后能正常工作

---

### 模式三：SAM3 文本提示标注

**目标**：使用 SAM3 的文本 grounding 能力，通过自然语言描述一次性找出图中所有匹配目标，批量确认后转为标注。

**使用模型**：SAM3（sam3.pt / sam3.1_multiplex.pt）
- 默认推荐：sam3.pt（标准模型）
- sam3.1_multiplex.pt 更重但支持更多并发目标

**交互流程**：
```
1. 切换到模式三（工具栏点击 SAM3 模式按钮）
2. 加载 SAM3 模型（首次加载，较慢，需显示加载进度）
3. 打开图片 → 自动调用 set_image() 预计算 vision 特征
4. 界面显示文本输入区域：
   ┌────────────────────────────────────────┐
   │ 文本提示: [________________] [搜索]    │
   │ 置信度阈值: [======▂▂▂▂] 0.5           │
   └────────────────────────────────────────┘
5. 用户输入文本（如"black cat"）→ 点击 [搜索] 或按 Enter
6. SAM3 调用 set_text_prompt() → 执行 grounding
7. 结果展示面板：
   ┌────────────────────────────────────────┐
   │ 找到 8 个候选（已过滤置信度 < 0.5）      │
   │                                        │
   │ ☑ Mask #1  [████████] 0.92 ┌────┐     │
   │ ☑ Mask #2  [██████  ] 0.87 │缩略│     │
   │ ☐ Mask #3  [████    ] 0.73 │图  │     │
   │ ☐ Mask #4  [███     ] 0.61 └────┘     │
   │ ...                                   │
   │                                        │
   │ [全选] [全不选] [反选]                  │
   │                                        │
   │ 选中 2 个 → [确认标注]                  │
   └────────────────────────────────────────┘
8. 用户可以：
   - 拖拽置信度滑块动态过滤候选
   - 勾选/取消勾选每个 mask
   - 点击某个 mask → Canvas 上高亮该 mask
   - 使用 [全选]/[全不选]/[反选] 批量操作
9. 可选：追加几何框提示（拖拽一个框）→ 调用 add_geometric_prompt() 修正
10. 点击 [确认标注] → 勾选的 masks 转为 polygons → 弹出标签选择
    → 所有选中 mask 使用同一标签批量添加
11. [重置] 按钮 → 清空文本提示和所有结果，可重新输入
```

**工具栏元素（模式三专属）**：
- 模型规格下拉：sam3 | sam3.1_multiplex
- 文本输入框 + [搜索] 按钮
- 置信度阈值滑块（默认 0.5，范围 0.0~1.0）
- [全选] [全不选] [反选] 按钮
- [确认标注] 按钮
- [重置] 按钮
- 候选 mask 列表（带勾选框、置信度条、缩略图）
- Canvas 上显示当前 hover/选中的候选 mask

**验收标准**：
- 能成功加载 SAM3 模型（至少 sam3.pt）
- 输入英文文本提示（如 "person", "cat", "car"）后能在 5 秒内返回候选 masks
- 候选列表按置信度降序排列
- 置信度滑块能实时过滤候选列表
- 勾选/取消勾选操作流畅
- 批量确认后所有选中 mask 正确转为 polygon 标注
- 追加框提示后结果能针对该区域修正
- 重置后状态完全清空
- 中文文本提示暂不要求（SAM3 官方训练数据以英文为主）

---

## 架构设计

### 目录结构

```
F:/数据标注工具/
├── DESIGN.md                    # 本设计文档
├── Models/                      # 模型权重（已存在）
│   ├── SAM2/
│   └── SAM3/
├── third_party_repository/      # 第三方仓库（已存在）
│   ├── sam2/
│   └── sam3/
├── src/                         # 工具源码
│   ├── main.py                  # 入口
│   ├── app.py                   # QApplication + 主窗口
│   ├── ui/
│   │   ├── main_window.py       # 主窗口布局
│   │   ├── canvas.py            # 图像 Canvas（绘制图片、标注、mask 预览）
│   │   ├── tool_bar.py          # 动态工具栏（随模式切换变化）
│   │   ├── label_panel.py       # 标签管理面板
│   │   ├── shape_list_panel.py  # 标注列表面板
│   │   ├── file_list_panel.py   # 文件浏览器面板
│   │   ├── model_control.py     # 模型选择与控制面板
│   │   └── sam3_text_panel.py   # SAM3 文本提示面板
│   ├── core/
│   │   ├── label_data.py        # 标注数据结构（兼容 labelme JSON）
│   │   ├── shape.py             # Shape 类（polygon, rectangle, etc.）
│   │   ├── io_manager.py        # JSON 读写
│   │   └── mask_utils.py        # mask→polygon 转换等工具函数
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_predictor.py    # 预测器抽象基类
│   │   ├── sam2_predictor.py    # SAM2 封装（模式二）
│   │   ├── sam3_predictor.py    # SAM3 封装（模式三：文本+几何）
│   │   └── model_manager.py     # 模型生命周期管理（加载/卸载/切换）
│   └── resources/
│       └── icons/               # 工具图标
├── tests/                       # 测试
│   ├── test_label_data.py
│   ├── test_mask_utils.py
│   ├── test_sam2_predictor.py
│   └── test_sam3_predictor.py
└── requirements.txt             # Python 依赖
```

### 核心类设计

```
BasePredictor (抽象基类)
├── is_loaded() -> bool
├── load_model(model_path) -> None
├── set_image(image: np.ndarray) -> None
├── reset() -> None
└── device() -> torch.device

SAM2Predictor(BasePredictor)
├── predict(points: List, labels: List, box: List) -> masks, scores
├── available_models() -> List[str]
└── current_model_name() -> str

SAM3Predictor(BasePredictor)
├── text_predict(text: str) -> List[CandidateMask]
├── add_box_prompt(box: List, label: bool)
├── set_confidence_threshold(t: float)
├── reset_prompts()
└── available_models() -> List[str]

CandidateMask (数据类)
├── mask: np.ndarray      # 二值 mask
├── score: float          # 置信度
├── bbox: List[float]     # 边界框 [x0,y0,x1,y1]
└── id: int               # 候选编号
```

### Canvas 层级绘制顺序

```
Layer 0: 原始图片
Layer 1: 已确认的标注形状（polygon 填充 + 边线）
Layer 2: 当前选中标注高亮
Layer 3: SAM 半透明 mask 预览（仅预览，非确认状态）
Layer 4: 提示点/框标记（绿点、红点、蓝框）
Layer 5: 正在绘制中的形状（如未闭合的多边形）
```

### 模式切换逻辑

```
切换模式时：
1. 如果当前有未确认的 mask 预览 → 提示用户确认或放弃
2. 切换工具栏显示内容
3. 切换 Canvas 的事件处理逻辑（鼠标行为随模式变化）
4. 模式三切换时还需要加载/卸载 SAM3 模型
```

## 数据流

```
┌──────────┐    np.ndarray     ┌──────────────┐    masks/scores    ┌──────────┐
│  Camera  │ ───────────────→  │  Predictor   │ ───────────────→  │  Canvas  │
│  /Disk   │    set_image()    │  (SAM2/SAM3) │   CandidateMask   │  显示预览  │
└──────────┘                   └──────────────┘                   └────┬─────┘
                                                                      │
                                                             用户确认/Enter
                                                                      │
                                                                      ▼
┌──────────┐    JSON 文件      ┌──────────────┐    Shape 对象    ┌──────────┐
│   磁盘    │ ←──────────────  │  IOManager   │ ←────────────  │  Shape   │
│          │    save/load     │              │   mask→polygon  │   List   │
└──────────┘                   └──────────────┘                 └──────────┘
```

## 依赖清单 (requirements.txt)

```
# UI
PySide6>=6.5

# Image processing
opencv-python>=4.8
Pillow>=10.0
numpy>=1.24

# ML (已有 data_annotation 环境)
torch>=2.0

# SAM2 依赖（从其 setup.py 推断）
# hydra-core, iopath, etc.（逐步补充）

# SAM3 依赖（从其 pyproject.toml 推断）
# huggingface_hub, torchvision, etc.（逐步补充）

# Utilities
shapely>=2.0        # polygon 简化/操作
```

## 实现阶段

### Phase 1：项目骨架 + 手动标注模式（模式一）
- 创建目录结构
- 搭建 PySide6 主窗口框架
- 实现 Canvas（图片加载、缩放、平移）
- 实现多边形工具（点击创建顶点、闭合、编辑顶点）
- 实现矩形工具
- 实现标签管理面板
- 实现标注列表面板
- 实现文件列表面板
- 实现 labelme JSON 读写
- 实现选择/删除/快捷键

### Phase 2：SAM2 集成（模式二）
- 编写 SAM2Predictor 封装
- 安装 SAM2 依赖到 conda 环境
- 实现模式二工具栏
- 实现 Canvas 上的点/框提示交互
- 实时 mask 预览（半透明叠加）
- mask→polygon 转换
- 迭代修正流程

### Phase 3：SAM3 文本模式（模式三）
- 编写 SAM3Predictor 封装
- 安装 SAM3 依赖到 conda 环境
- 实现 SAM3 文本输入面板
- 实现候选 mask 列表（勾选、缩略图、置信度条）
- Canvas 上候选 mask 可视化
- 置信度滑块动态过滤
- 批量确认+标签分配

### Phase 4：打磨
- 样式美化（QSS）
- 性能优化（大图加载、mask 渲染缓存）
- 错误处理（模型加载失败、显存不足等场景的友好提示）
- 综合测试

## 验收总标准

1. **模式一**：不加载任何模型，能独立完成手动标注全流程（画→标→存→再打开）
2. **模式二**：加载 SAM2 后，3 次点击以内能生成质量可接受的 mask，确认后转为 polygon 标注
3. **模式三**：输入文本描述后能返回候选 masks，勾选确认后批量生成标注
4. **兼容性**：所有模式生成的 JSON 文件可被 labelme 打开和编辑
5. **稳定性**：连续标注 20 张图片不崩溃、不出现明显内存泄漏
6. **可用性**：模式切换流畅，模型加载有进度提示，错误有中文提示
