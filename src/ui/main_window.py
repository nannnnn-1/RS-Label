from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.io_manager import IOManager
from ..core.label_data import LabelData
from .canvas import AnnotationCanvas, CanvasMode, ToolType
from .file_list_panel import FileListPanel
from .label_panel import LabelPanel
from .sam3_panel import SAM3Panel
from .shape_list_panel import ShapeListPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 数据标注工具")
        self.resize(1400, 900)

        self._label_data: Optional[LabelData] = None
        self._save_path: str = ""
        self._dirty: bool = False
        self._sam2_predictor = None
        self._sam3_predictor = None

        self._setup_ui()
        self._setup_shortcuts()
        self._connect_signals()

        # Set initial mode (must be after _canvas is created)
        self._on_mode_manual()

        # Apply dark theme
        self._apply_dark_theme()

    def _setup_ui(self):
        # --- Menu Bar ---
        menu = self.menuBar()

        file_menu = menu.addMenu("文件(&F)")
        file_menu.addAction("打开图片(&O)...", self._open_image, QKeySequence.Open)
        file_menu.addAction("打开目录(&D)...", self._open_directory, QKeySequence("Ctrl+D"))
        file_menu.addSeparator()
        file_menu.addAction("保存(&S)", self._save, QKeySequence.Save)
        file_menu.addAction("另存为(&A)...", self._save_as, QKeySequence("Ctrl+Shift+S"))
        file_menu.addSeparator()
        file_menu.addAction("上一张", self._prev_image, QKeySequence("A"))
        file_menu.addAction("下一张", self._next_image, QKeySequence("D"))
        file_menu.addSeparator()
        file_menu.addAction("退出(&Q)", self.close, QKeySequence.Quit)

        edit_menu = menu.addMenu("编辑(&E)")
        edit_menu.addAction("删除选中标注", self._delete_selected, QKeySequence.Delete)
        edit_menu.addAction("撤销", self._undo, QKeySequence.Undo)

        view_menu = menu.addMenu("视图(&V)")
        view_menu.addAction("适应窗口", self._fit_view, QKeySequence("F"))
        view_menu.addAction("放大", self._zoom_in, QKeySequence.ZoomIn)
        view_menu.addAction("缩小", self._zoom_out, QKeySequence.ZoomOut)

        help_menu = menu.addMenu("帮助(&H)")
        help_menu.addAction("关于", self._about)

        # --- Toolbar ---
        self._toolbar = QToolBar("工具栏")
        self._toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self._toolbar)

        # Mode switching
        self._toolbar.addWidget(QLabel(" 模式: "))
        self._btn_mode_manual = self._make_tool_btn("手动", self._on_mode_manual, "1")
        self._btn_mode_sam2 = self._make_tool_btn("SAM2", self._on_mode_sam2, "2")
        self._btn_mode_sam3 = self._make_tool_btn("SAM3", self._on_mode_sam3, "3")
        self._toolbar.addWidget(self._btn_mode_manual)
        self._toolbar.addWidget(self._btn_mode_sam2)
        self._toolbar.addWidget(self._btn_mode_sam3)
        self._toolbar.addSeparator()

        # Manual tool buttons (visible in manual mode)
        self._manual_tools = []
        self._btn_select = self._make_tool_btn("选择", self._on_select_tool, "S")
        self._btn_polygon = self._make_tool_btn("多边形", self._on_polygon_tool, "P")
        self._btn_rectangle = self._make_tool_btn("矩形", self._on_rectangle_tool, "R")
        for btn in [self._btn_select, self._btn_polygon, self._btn_rectangle]:
            self._toolbar.addWidget(btn)
            self._manual_tools.append(btn)

        self._toolbar.addSeparator()

        # SAM2 controls (visible in SAM2 mode, hidden in manual)
        self._sam2_widgets = []
        self._toolbar.addWidget(QLabel(" SAM2:"))
        self._sam2_model_combo = QComboBox()
        self._sam2_model_combo.setMinimumWidth(90)
        self._sam2_model_combo.addItems(["base_plus", "tiny", "small", "large"])
        self._sam2_model_combo.setCurrentText("base_plus")
        self._toolbar.addWidget(self._sam2_model_combo)
        self._sam2_widgets.append(self._sam2_model_combo)

        self._sam2_load_btn = QPushButton("加载")
        self._sam2_load_btn.setFixedWidth(50)
        self._sam2_load_btn.clicked.connect(self._load_sam2_model)
        self._toolbar.addWidget(self._sam2_load_btn)
        self._sam2_widgets.append(self._sam2_load_btn)

        self._sam2_status = QLabel(" 未加载")
        self._sam2_status.setStyleSheet("color: #888; font-size: 11px;")
        self._toolbar.addWidget(self._sam2_status)
        self._sam2_widgets.append(self._sam2_status)

        # SAM3 controls (visible in SAM3 mode)
        self._sam3_widgets = []
        self._toolbar.addWidget(QLabel(" SAM3:"))
        self._sam3_model_combo = QComboBox()
        self._sam3_model_combo.setMinimumWidth(70)
        self._sam3_model_combo.addItems(["sam3", "sam3.1"])
        self._sam3_model_combo.setCurrentText("sam3")
        self._toolbar.addWidget(self._sam3_model_combo)
        self._sam3_widgets.append(self._sam3_model_combo)

        self._sam3_load_btn = QPushButton("加载")
        self._sam3_load_btn.setFixedWidth(50)
        self._sam3_load_btn.clicked.connect(self._load_sam3_model)
        self._toolbar.addWidget(self._sam3_load_btn)
        self._sam3_widgets.append(self._sam3_load_btn)

        self._sam3_status = QLabel(" 未加载")
        self._sam3_status.setStyleSheet("color: #888; font-size: 11px;")
        self._toolbar.addWidget(self._sam3_status)
        self._sam3_widgets.append(self._sam3_status)

        self._toolbar.addSeparator()

        # Label display
        self._toolbar.addWidget(QLabel(" 标签: "))
        self._current_label_widget = QLabel("(未选择)")
        self._current_label_widget.setStyleSheet(
            "font-weight: bold; color: #4ECDC4; padding: 2px 8px;"
            "background: #333; border-radius: 3px;"
        )
        self._toolbar.addWidget(self._current_label_widget)
        self._toolbar.addSeparator()

        # --- Central Area ---
        splitter = QSplitter(Qt.Horizontal)

        # Left: file list
        self._file_panel = FileListPanel()
        splitter.addWidget(self._file_panel)

        # Center: canvas
        self._canvas = AnnotationCanvas()
        splitter.addWidget(self._canvas)

        # Right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._label_panel = LabelPanel()
        right_layout.addWidget(self._label_panel)

        self._shape_panel = ShapeListPanel()
        right_layout.addWidget(self._shape_panel)

        self._sam3_panel = SAM3Panel()
        self._sam3_panel.setVisible(False)
        right_layout.addWidget(self._sam3_panel)

        right_layout.addStretch()
        splitter.addWidget(right_widget)

        splitter.setSizes([200, 900, 250])
        self.setCentralWidget(splitter)

        # --- Status Bar ---
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("就绪 - 请打开图片目录开始标注")

    def _make_tool_btn(self, text, callback=None, shortcut=""):
        btn = QToolButton()
        btn.setText(text)
        btn.setCheckable(True)
        if callback:
            btn.clicked.connect(callback)
        if shortcut:
            btn.setToolTip(f"{text} ({shortcut})")
        return btn

    def _setup_shortcuts(self):
        pass  # Handled via QAction shortcuts and keyPressEvent on canvas

    def _connect_signals(self):
        self._canvas.shape_added.connect(self._on_shape_added)
        self._canvas.shape_edited.connect(self._on_shape_edited)
        self._canvas.shape_deleted.connect(self._on_shape_deleted)
        self._canvas.status_message.connect(self._status.showMessage)
        self._canvas.image_loaded.connect(self._on_image_loaded)

        self._label_panel.label_added.connect(self._on_label_added)
        self._label_panel.label_removed.connect(self._on_label_removed)
        self._label_panel.label_selected.connect(self._on_label_selected)

        self._shape_panel.shape_clicked.connect(self._on_shape_list_clicked)
        self._shape_panel.shape_deleted.connect(self._on_shape_list_deleted)

        self._file_panel.file_selected.connect(self._on_file_selected)
        self._file_panel.directory_changed.connect(self._on_directory_changed)

        self._sam3_panel.text_search_requested.connect(self._on_sam3_search)
        self._sam3_panel.threshold_changed.connect(self._on_sam3_threshold)
        self._sam3_panel.candidates_confirmed.connect(self._on_sam3_confirm)
        self._sam3_panel.candidate_hovered.connect(self._canvas.highlight_sam3_candidate)

    # --- Actions ---
    def _open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp);;所有文件 (*)"
        )
        if path:
            self._load_image_file(path)

    def _open_directory(self):
        self._file_panel.open_directory()

    def _load_image_file(self, image_path: str):
        if not self._maybe_save():
            return
        try:
            self._canvas.load_image(image_path)
        except Exception as e:
            self._status.showMessage(f"图片加载失败: {e}")
            return

        self._label_data = self._canvas.get_label_data()
        self._save_path = ""

        # Check for existing label file
        label_path = IOManager.get_label_path(image_path)
        if os.path.exists(label_path):
            try:
                label_data = IOManager.load_label_file(label_path)
                self._canvas.set_label_data(label_data)
                self._label_data = label_data
                self._save_path = label_path
                self._status.showMessage(f"已加载已有标注: {label_path}")
            except Exception as e:
                self._status.showMessage(f"标注文件加载失败: {e}")

        self._sync_labels_from_data()
        self._update_panels()
        self._dirty = False

    def _save(self):
        if not self._label_data:
            self._status.showMessage("无标注数据可保存")
            return
        if not self._save_path:
            img_path = self._canvas.image_path()
            if not img_path:
                self._status.showMessage("请先打开图片")
                return
            self._save_path = IOManager.get_label_path(img_path)

        self._sync_data_from_canvas()
        try:
            IOManager.save_label_file(self._label_data, self._save_path)
            self._dirty = False
            self._status.showMessage(f"已保存: {self._save_path}")
        except Exception as e:
            self._status.showMessage(f"保存失败: {e}")

    def _save_as(self):
        if not self._label_data:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "另存为", IOManager.get_label_path(self._canvas.image_path()),
            "JSON 文件 (*.json)"
        )
        if path:
            self._save_path = path
            self._save()

    def _prev_image(self):
        self._file_panel.navigate_prev()

    def _next_image(self):
        self._file_panel.navigate_next()

    def _delete_selected(self):
        idx = self._canvas.selected_shape_index()
        if idx >= 0:
            del self._label_data.shapes[idx]
            self._canvas._selected_shape_idx = -1
            self._canvas._update_shape_items()
            self._update_panels()
            self._dirty = True

    def _undo(self):
        if self._label_data and self._label_data.shapes:
            self._label_data.shapes.pop()
            self._canvas._selected_shape_idx = -1
            self._canvas._update_shape_items()
            self._update_panels()
            self._dirty = True

    def _fit_view(self):
        self._canvas.fitInView(
            self._canvas.scene().sceneRect(), Qt.KeepAspectRatio
        )

    def _zoom_in(self):
        self._canvas.scale(1.25, 1.25)

    def _zoom_out(self):
        self._canvas.scale(0.8, 0.8)

    def _about(self):
        QMessageBox.about(self, "关于", "SAM 半自动数据标注工具\n\n"
                          "模式一：手动标注（多边形/矩形）\n"
                          "模式二：SAM2 点/框提示标注\n"
                          "模式三：SAM3 文本提示标注")

    # --- Tool switching ---
    def _on_select_tool(self):
        self._update_tool_buttons(self._btn_select)
        self._canvas.set_tool(ToolType.SELECT)

    def _on_polygon_tool(self):
        self._update_tool_buttons(self._btn_polygon)
        self._canvas.set_tool(ToolType.POLYGON)

    def _on_rectangle_tool(self):
        self._update_tool_buttons(self._btn_rectangle)
        self._canvas.set_tool(ToolType.RECTANGLE)

    def _update_tool_buttons(self, active):
        for btn in [self._btn_select, self._btn_polygon, self._btn_rectangle]:
            btn.setChecked(btn is active)

    # --- Mode switching ---

    def _on_mode_manual(self):
        self._update_mode_buttons(self._btn_mode_manual)
        self._canvas.set_mode(CanvasMode.MANUAL)
        self._canvas.clear_sam3_previews()
        for w in self._manual_tools:
            w.setEnabled(True)
        for w in self._sam2_widgets:
            w.setEnabled(False)
        for w in self._sam3_widgets:
            w.setEnabled(False)
        self._sam3_panel.hide()
        self._status.showMessage("模式一：手动标注")

    def _on_mode_sam2(self):
        self._update_mode_buttons(self._btn_mode_sam2)
        self._canvas.set_mode(CanvasMode.SAM2)
        for w in self._manual_tools:
            w.setEnabled(False)
        for w in self._sam2_widgets:
            w.setEnabled(True)
        for w in self._sam3_widgets:
            w.setEnabled(False)
        self._sam3_panel.hide()
        self._status.showMessage("模式二：SAM2 交互式标注 (正点=左键 负点=右键 Ctrl+拖=框 Enter=确认 Esc=取消)")

    def _on_mode_sam3(self):
        self._update_mode_buttons(self._btn_mode_sam3)
        self._canvas.set_mode(CanvasMode.SAM3)
        for w in self._manual_tools:
            w.setEnabled(False)
        for w in self._sam2_widgets:
            w.setEnabled(False)
        for w in self._sam3_widgets:
            w.setEnabled(True)
        self._sam3_panel.show()
        self._status.showMessage("模式三：SAM3 文本提示标注")

    def _update_mode_buttons(self, active):
        for btn in [self._btn_mode_manual, self._btn_mode_sam2, self._btn_mode_sam3]:
            btn.setChecked(btn is active)

    # --- SAM2 model management ---

    def _load_sam2_model(self):
        model_name = self._sam2_model_combo.currentText()
        self._status.showMessage(f"正在加载 SAM2 {model_name}...")
        self._sam2_load_btn.setEnabled(False)

        if self._sam2_predictor is None:
            from ..models.sam2_predictor import SAM2Predictor
            self._sam2_predictor = SAM2Predictor()

        try:
            msg = self._sam2_predictor.load_model(model_name)
            self._canvas.set_sam2_predictor(self._sam2_predictor)
            self._sam2_status.setText(f" {model_name} ✓")
            self._sam2_status.setStyleSheet(
                "color: #4ECDC4; font-size: 11px; font-weight: bold;"
            )
            self._status.showMessage(msg)
        except Exception as e:
            self._sam2_status.setText(" 加载失败")
            self._sam2_status.setStyleSheet("color: #FF6B6B; font-size: 11px;")
            self._status.showMessage(f"加载失败: {e}")
        finally:
            self._sam2_load_btn.setEnabled(True)

    def _load_sam3_model(self):
        model_name = self._sam3_model_combo.currentText()
        self._status.showMessage(f"正在加载 SAM3 {model_name}...")
        self._sam3_load_btn.setEnabled(False)

        if self._sam3_predictor is None:
            from ..models.sam3_predictor import SAM3Predictor
            self._sam3_predictor = SAM3Predictor()

        try:
            msg = self._sam3_predictor.load_model(model_name)
            self._sam3_status.setText(f" {model_name} ✓")
            self._sam3_status.setStyleSheet(
                "color: #4ECDC4; font-size: 11px; font-weight: bold;"
            )
            self._status.showMessage(msg)
        except Exception as e:
            self._sam3_status.setText(" 加载失败")
            self._sam3_status.setStyleSheet("color: #FF6B6B; font-size: 11px;")
            self._status.showMessage(f"加载失败: {e}")
        finally:
            self._sam3_load_btn.setEnabled(True)

    # --- SAM3 handlers ---

    def _on_sam3_search(self, text: str):
        if self._sam3_predictor is None or not self._sam3_predictor.is_loaded():
            self._status.showMessage("请先加载 SAM3 模型")
            return
        if self._canvas._numpy_image is None:
            self._status.showMessage("请先打开图片")
            return

        try:
            self._sam3_predictor.set_image(self._canvas._numpy_image)
            results = self._sam3_predictor.text_predict(text)
            self._sam3_panel.set_candidates(results)

            # Update canvas preview with all candidate masks
            threshold = self._sam3_predictor._processor.confidence_threshold
            masks = [(r["mask"], r["score"]) for r in results if r["score"] >= threshold]
            self._canvas.show_sam3_mask_preview(masks)
            self._status.showMessage(f"SAM3 找到 {len(results)} 个候选")
        except Exception as e:
            self._status.showMessage(f"SAM3 搜索失败: {e}")

    def _on_sam3_threshold(self, value: float):
        if self._sam3_predictor is not None:
            self._sam3_predictor.set_confidence_threshold(value)
            # Re-run search with new threshold
            text = self._sam3_panel._text_input.text().strip()
            if text:
                self._on_sam3_search(text)

    def _on_sam3_confirm(self, candidates):
        if not self._label_data:
            return
        from ..core.mask_utils import largest_polygon
        label = self._canvas._current_label
        added = 0
        for c in candidates:
            poly = largest_polygon(c.mask)
            if poly:
                from ..core.shape import Shape, ShapeType
                shape = Shape(label=label, points=poly, shape_type=ShapeType.POLYGON)
                self._label_data.shapes.append(shape)
                added += 1

        if added > 0:
            self._canvas._update_shape_items()
            self._update_panels()
            self._dirty = True
            self._status.showMessage(f"SAM3: 已确认 {added} 个标注")
            self._canvas.clear_sam3_previews()
            self._sam3_panel.clear()

    # --- Signal handlers ---
    def _on_shape_added(self, idx: int):
        self._update_panels()
        self._dirty = True

    def _on_shape_edited(self, idx: int):
        self._update_panels()
        self._dirty = True

    def _on_shape_deleted(self, idx: int):
        self._update_panels()
        self._dirty = True

    def _on_image_loaded(self, path: str):
        pass

    def _on_label_added(self, label: str):
        pass

    def _on_label_removed(self, label: str):
        pass

    def _on_label_selected(self, label: str):
        self._canvas.set_current_label(label)
        self._current_label_widget.setText(label)

    def _on_shape_list_clicked(self, idx: int):
        self._canvas._selected_shape_idx = idx
        self._canvas._update_shape_items()

    def _on_shape_list_deleted(self, idx: int):
        if 0 <= idx < len(self._label_data.shapes):
            del self._label_data.shapes[idx]
            self._canvas._selected_shape_idx = -1
            self._canvas._update_shape_items()
            self._update_panels()
            self._dirty = True

    def _on_file_selected(self, path: str):
        if path != self._canvas.image_path():
            self._maybe_save()
            self._load_image_file(path)

    def _on_directory_changed(self, dir_path: str):
        self.setWindowTitle(f"SAM 数据标注工具 - {dir_path}")

    # --- Helpers ---
    def _update_panels(self):
        if self._label_data:
            self._shape_panel.update_shapes(self._label_data.shapes)
            # Merge: keep existing labels + add any new ones from current shapes
            existing = set(self._label_panel.labels())
            current = set(s.label for s in self._label_data.shapes)
            labels = list(dict.fromkeys(
                list(self._label_panel.labels()) + list(current)
            ))
            self._label_panel.set_labels(labels)

    def _sync_labels_from_data(self):
        if self._label_data:
            # Merge current image labels into existing panel labels
            existing = set(self._label_panel.labels())
            current = set(s.label for s in self._label_data.shapes)
            merged = list(dict.fromkeys(
                list(self._label_panel.labels()) + list(current)
            ))
            self._label_panel.set_labels(merged)

    def _sync_data_from_canvas(self):
        self._label_data = self._canvas.get_label_data()

    def _maybe_save(self) -> bool:
        if self._dirty and self._label_data and self._label_data.shapes:
            reply = QMessageBox.question(
                self, "未保存的更改",
                "当前图片有未保存的标注，是否保存？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if reply == QMessageBox.Save:
                self._save()
                return True
            elif reply == QMessageBox.Cancel:
                return False
        return True

    def closeEvent(self, event):
        if self._maybe_save():
            event.accept()
        else:
            event.ignore()

    def _apply_dark_theme(self):
        app = QApplication.instance()
        if app:
            app.setStyle("Fusion")
            palette = QPalette()
            palette.setColor(QPalette.Window, QColor("#2b2b2b"))
            palette.setColor(QPalette.WindowText, QColor("#e0e0e0"))
            palette.setColor(QPalette.Base, QColor("#1e1e1e"))
            palette.setColor(QPalette.AlternateBase, QColor("#2b2b2b"))
            palette.setColor(QPalette.ToolTipBase, QColor("#3b3b3b"))
            palette.setColor(QPalette.ToolTipText, QColor("#e0e0e0"))
            palette.setColor(QPalette.Text, QColor("#e0e0e0"))
            palette.setColor(QPalette.Button, QColor("#3b3b3b"))
            palette.setColor(QPalette.ButtonText, QColor("#e0e0e0"))
            palette.setColor(QPalette.BrightText, QColor("#ff5555"))
            palette.setColor(QPalette.Link, QColor("#4ECDC4"))
            palette.setColor(QPalette.Highlight, QColor("#4ECDC4"))
            palette.setColor(QPalette.HighlightedText, QColor("#1e1e1e"))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor("#666666"))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#666666"))
            app.setPalette(palette)
