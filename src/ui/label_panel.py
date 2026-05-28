from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.shape import get_label_color


class LabelPanel(QWidget):
    label_added = Signal(str)
    label_removed = Signal(str)
    label_selected = Signal(str)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._labels: list[str] = []
        self._selected: str = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("标签列表")
        header.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(header)

        # Label list
        self._list = QListWidget()
        self._list.setMaximumHeight(250)
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ 新建")
        add_btn.clicked.connect(self._add_label)
        del_btn = QPushButton("- 删除")
        del_btn.clicked.connect(self._remove_label)
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(del_btn)
        layout.addLayout(btn_layout)

        self.setMinimumWidth(160)

    def labels(self) -> list[str]:
        return self._labels[:]

    def selected_label(self) -> str:
        return self._selected

    def set_labels(self, labels: list[str]):
        self._labels = list(dict.fromkeys(labels))  # dedupe preserve order
        self._refresh_list()

    def _add_label(self):
        text, ok = QInputDialog.getText(
            self, "新建标签", "标签名称:"
        )
        if ok and text.strip():
            text = text.strip()
            if text not in self._labels:
                self._labels.append(text)
                self._refresh_list()
                self.label_added.emit(text)

    def _remove_label(self):
        if self._selected and self._selected in self._labels:
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除标签 '{self._selected}' 吗？\n已有的标注不会被删除。",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self._labels.remove(self._selected)
                self._selected = ""
                self._refresh_list()
                self.label_removed.emit(self._selected)

    def _on_item_clicked(self, item: QListWidgetItem):
        self._selected = item.text()
        self.label_selected.emit(self._selected)

    def _refresh_list(self):
        self._list.clear()
        for i, label in enumerate(self._labels):
            item = QListWidgetItem(label)
            color = get_label_color(label, i)
            item.setForeground(color)
            self._list.addItem(item)
