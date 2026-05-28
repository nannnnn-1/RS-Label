from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class FileListPanel(QWidget):
    file_selected = Signal(str)
    directory_changed = Signal(str)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("文件列表")
        header.setFont(QFont("Arial", 11, QFont.Bold))
        layout.addWidget(header)

        # Directory controls
        dir_layout = QHBoxLayout()
        self._dir_label = QLabel("未选择目录")
        self._dir_label.setWordWrap(True)
        self._dir_label.setStyleSheet("color: #888; font-size: 10px;")
        dir_layout.addWidget(self._dir_label)

        browse_btn = QPushButton("...")
        browse_btn.setFixedWidth(30)
        browse_btn.clicked.connect(self._browse_directory)
        dir_layout.addWidget(browse_btn)
        layout.addLayout(dir_layout)

        # File list
        self._list = QListWidget()
        self._list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._list)

        # Stats
        self._stats = QLabel("")
        self._stats.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._stats)

        self._current_dir: str = ""
        self._files: list[str] = []
        self._current_index: int = -1

        self.setMinimumWidth(180)
        self.setMaximumWidth(280)

    def open_directory(self, dir_path: str = None):
        if dir_path is None:
            dir_path = QFileDialog.getExistingDirectory(
                self, "选择图片目录", self._current_dir or os.path.expanduser("~")
            )
        if not dir_path:
            return

        self._current_dir = dir_path
        self._dir_label.setText(dir_path)
        self._refresh_files()
        self.directory_changed.emit(dir_path)

    def current_directory(self) -> str:
        return self._current_dir

    def current_file(self) -> str:
        if 0 <= self._current_index < len(self._files):
            return self._files[self._current_index]
        return ""

    def navigate_next(self) -> str:
        if not self._files:
            return ""
        self._current_index = min(self._current_index + 1, len(self._files) - 1)
        return self._navigate_to_current()

    def navigate_prev(self) -> str:
        if not self._files:
            return ""
        self._current_index = max(self._current_index - 1, 0)
        return self._navigate_to_current()

    def _navigate_to_current(self) -> str:
        filepath = self.current_file()
        if filepath:
            self._list.setCurrentRow(self._current_index)
            self.file_selected.emit(filepath)
        return filepath

    def _browse_directory(self):
        self.open_directory()

    def _refresh_files(self):
        self._list.clear()
        self._files = []

        if not self._current_dir:
            return

        for f in sorted(os.listdir(self._current_dir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                full_path = os.path.join(self._current_dir, f)
                self._files.append(full_path)
                item = QListWidgetItem(f)
                self._list.addItem(item)

        self._stats.setText(f"{len(self._files)} 张图片")
        if self._files:
            self._current_index = 0
            self._list.setCurrentRow(0)

    def _on_item_clicked(self, item: QListWidgetItem):
        self._current_index = self._list.row(item)
        filepath = self.current_file()
        if filepath:
            self.file_selected.emit(filepath)
