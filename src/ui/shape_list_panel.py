from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.shape import Shape, get_label_color


class ShapeListPanel(QWidget):
    shape_clicked = Signal(int)    # index
    shape_deleted = Signal(int)    # index
    shape_label_changed = Signal(int, str)  # index, new_label

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._shapes: list[Shape] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("标注列表")
        header.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(header)

        self._list = QListWidget()
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        # Stats
        self._stats = QLabel("")
        self._stats.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._stats)

        self.setMinimumWidth(160)

    def update_shapes(self, shapes: list[Shape]):
        self._shapes = shapes
        self._refresh_list()

    def _refresh_list(self):
        self._list.clear()
        for i, shape in enumerate(self._shapes):
            color = get_label_color(shape.label, i)
            pts_count = len(shape.points)
            display_label = shape.label or "(未命名)"
            text = f"[{display_label}] {shape.shape_type.value} ({pts_count}点)"
            item = QListWidgetItem(text)
            item.setForeground(color)
            self._list.addItem(item)
        self._stats.setText(f"共 {len(self._shapes)} 个标注")

    def _on_item_clicked(self, item: QListWidgetItem):
        idx = self._list.row(item)
        self.shape_clicked.emit(idx)

    def select_index(self, idx: int):
        if 0 <= idx < self._list.count():
            self._list.setCurrentRow(idx)
