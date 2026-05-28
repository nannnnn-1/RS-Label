# 开发接手文档

## 项目概览

SAM 半自动数据标注工具，基于 PySide6 + SAM2/SAM3，三种标注模式，输出 labelme 兼容 JSON。

- **仓库**: https://github.com/nannnnn-1/RS-Label.git
- **Python**: 3.12
- **Conda 环境**: data_annotation
- **关键硬件**: NVIDIA GPU（6GB+ 显存），SAM3 需要 6GB+

## 目录结构

```
RS-Label/
├── run.py              # 启动入口: python run.py
├── requirements.txt    # pip 依赖
├── DESIGN.md           # 完整设计方案 + 验收标准
├── USAGE.md            # 用户使用指南
├── HANDOVER.md         # 本文件
├── src/
│   ├── main.py         # 备用入口
│   ├── core/           # 数据模型 (与UI无关)
│   │   ├── shape.py        # Shape 数据类, 20色标签色板, get_label_color()
│   │   ├── label_data.py   # LabelData → labelme JSON 格式
│   │   ├── io_manager.py   # 图片加载, JSON 读写
│   │   └── mask_utils.py   # mask→polygon (cv2.findContours + approxPolyDP)
│   ├── models/         # 模型封装 (与UI无关)
│   │   ├── base_predictor.py   # 抽象基类: load/set_image/predict/unload
│   │   ├── sam2_predictor.py   # SAM2: OmegaConf 直接加载配置(绕过Hydra中文路径bug)
│   │   └── sam3_predictor.py   # SAM3: 含 bfloat16 dtype 修补
│   └── ui/             # PySide6 前端
│       ├── main_window.py      # 主窗口: 三模式切换, 暗色主题, 菜单/快捷键
│       ├── canvas.py           # QGraphicsView 画布: 三工具 + SAM2/SAM3预览
│       ├── sam2_tool.py        # SAM2 点/框交互: 左键正点, 右键负点, Ctrl+拖框
│       ├── sam3_panel.py       # SAM3 文本面板: 输入+候选列表+置信度滑块
│       ├── label_panel.py      # 标签管理: 增删, 标签色
│       ├── shape_list_panel.py # 标注列表: 显示所有shape, 点击选中
│       └── file_list_panel.py  # 文件浏览: 目录+图片列表
├── tests/
│   ├── test_core.py       # 7项核心测试(Shape/LabelData/JSON/mask)
│   └── test_stability.py  # 22张图压力测试 + SAM2重载 + SAM3
├── Models/                # 不在git中, 需手动拷贝
│   ├── SAM2/  (*.pt, 149M~857M)
│   └── SAM3/  (sam3.pt 3.3G, sam3.1_multiplex.pt 3.3G)
└── third_party_repository/ # 不在git中, 需重新clone
    ├── sam2/   (facebookresearch/sam2)
    └── sam3/   (facebookresearch/sam3)
```

## 关键技术决策

### 1. SAM2 模型加载: 绕过 Hydra 中文路径问题

**问题**: `F:\数据标注工具\...` 路径含中文，Hydra 的 `compose()` 无法找到配置文件。

**解决方案**: 不调用 `build_sam` 中的 `compose()`，改为直接用 OmegaConf 加载 YAML:
```python
cfg = OmegaConf.load(config_path)  # 直接路径, 不经过Hydra search path
model = instantiate(cfg.model, _recursive_=True)
```
见 `src/models/sam2_predictor.py` 第 60 行附近。

### 2. SAM3 dtype 问题

**问题**: SAM3 的 `sam3/perflib/fused.py` 中 `addmm_act()` 函数会把中间激活转为 bfloat16，但 PyTorch 2.12 严格检查 matmul 的 dtype 一致，导致 `mat1 and mat2 must have the same dtype, but got BFloat16 and Float`。

**解决方案**: 在 `build_sam3_image_model` 导入完成后、模型构建前，monkey-patch `addmm_act`，让输出转回权重的原始 dtype:
```python
def patched(activation, linear, mat1):
    out = _original(activation, linear, mat1)
    return out.to(linear.weight.dtype)
```
见 `src/models/sam3_predictor.py` 的 `_patch_dtype()` 方法。

### 3. Windows Triton 依赖

**问题**: SAM3 依赖 `triton`（Linux only）用于视频追踪的 EDT 计算。

**解决方案**: Windows 上安装 `triton-windows==3.3.0.post19`。图片标注模式不走 EDT 代码路径，所以不会触发实际调用。见 `requirements.txt`。

### 4. Qt6 兼容性

- `QPointF` 没有 `.toPoint()` 方法（Qt5有，Qt6移除）→ `canvas.py` 中去掉了所有 `.toPoint()` 调用
- `AA_EnableHighDpiScaling` / `AA_UseHighDpiPixmaps` 在 Qt6 已废弃 → `run.py` 中已移除
- `QPalette.Window` 存在但 color role 名称是小写 `window` → 使用 `QPalette.Window` 正常
- QToolBar 不支持 `hide()/show()` 子 widget → 改用 `setEnabled(True/False)` 切换模式

### 5. Polygon 简化精度

**问题**: 初始 `epsilon_factor=0.005`（周长的 0.5%）导致 SAM mask 转 polygon 后过于简陋。

**修复**: 改为 `epsilon_factor=0.001`（周长的 0.1%），顶点数提升约 8 倍。见 `src/core/mask_utils.py`。

### 6. 标签跨图持久化

**问题**: `_sync_labels_from_data` 和 `_update_panels` 用当前图 shapes 的标签覆盖面板，切到无标注的新图时清空标签列表。

**修复**: 改为合并策略——在已有标签基础上追加当前图的标签，不删除。标签列表内容来自内存，关工具后不保留（除非从已有 JSON 恢复）。

### 7. 导入路径

`run.py` 必须添加项目根目录到 `sys.path`（不是 `src/` 目录），然后 `from src.ui.main_window import MainWindow`。原因是 `src/` 内使用相对导入 `from ..core`，需要 `src` 作为包根。

## 数据流

```
用户操作 → Canvas(工具) → Predictor.set_image() → 模型推理
                                    ↓
              Canvas ← mask预览 ← Predictor.predict()
                 ↓ Enter确认
           mask_utils.mask_to_polygons() → Shape → LabelData.shapes
                 ↓ Ctrl+S
           IOManager.save_label_file() → .json (labelme格式)
```

## 已知问题 & 待改进

1. **SAM3 模型加载慢**: 3.3GB 模型，首次加载约 15-30 秒。考虑加 loading 动画。
2. **SAM3 候选重复过滤**: 当前 `_on_sam3_threshold` 每次调都会重新推理，应改为前端过滤。
3. **mask 预览性能**: 大图（4K+）的 mask overlay 渲染可能卡顿，需要降采样或 GPU 渲染。
4. **SAM3 仅英文**: 文本提示建议使用英文描述。
5. **GPU 显存**: SAM2(base_plus) ~617MB, SAM3 ~6GB。两者同时加载可能超出 6GB 显存。当前同一时间只能加载一个。
6. **手动标注撤销**: 当前只支持撤销最后一个 shape，不支持多步 undo。

## 测试

```bash
# 核心数据模型测试
python tests/test_core.py

# 稳定性测试 (22张图)
python tests/test_stability.py
```

## 常见开发操作

### 添加新的模型变体

编辑 `src/models/sam2_predictor.py` 的 `MODEL_VARIANTS` 字典，添加 config/checkpoint 映射。

### 修改快捷键

`src/ui/main_window.py` → `_setup_ui()` 的菜单部分，以及 `canvas.py` 的 `keyPressEvent`。

### 修改标签颜色

`src/core/shape.py` → `LABEL_COLORS` 列表（20 种颜色循环使用）。

### 调试 SAM2 预测

```python
from src.models.sam2_predictor import SAM2Predictor
p = SAM2Predictor()
p.load_model("base_plus")
import numpy as np
p.set_image(your_image_array)
masks, scores = p.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),
)
```

### 调试 SAM3 预测

```python
from src.models.sam3_predictor import SAM3Predictor
p = SAM3Predictor()
p.load_model("sam3")
p.set_image(your_image_array)
results = p.text_predict("a cat")
# results: [{"mask": np.ndarray, "score": float, "bbox": [x0,y0,x1,y1]}, ...]
```
