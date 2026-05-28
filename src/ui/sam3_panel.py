from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.shape import get_label_color


class CandidateItem:
    """Data holder for a single SAM3 candidate."""

    def __init__(self, mask: np.ndarray, score: float, bbox: list, index: int):
        self.mask = mask
        self.score = score
        self.bbox = bbox
        self.index = index
        self.checked = True


class SAM3Panel(QWidget):
    text_search_requested = Signal(str)
    threshold_changed = Signal(float)
    candidates_confirmed = Signal(list)
    candidate_hovered = Signal(int)   # index of hovered candidate, -1 to clear

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._candidates: List[CandidateItem] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Header
        hdr = QLabel("SAM3 文本提示")
        hdr.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(hdr)

        # Text input
        input_layout = QHBoxLayout()
        self._text_input = QLineEdit()
        self._text_input.setPlaceholderText("输入英文描述，如: a black cat")
        self._text_input.returnPressed.connect(self._on_search)
        input_layout.addWidget(self._text_input)

        search_btn = QPushButton("搜索")
        search_btn.clicked.connect(self._on_search)
        input_layout.addWidget(search_btn)
        layout.addLayout(input_layout)

        # Confidence threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("置信度阈值:"))

        self._threshold_slider = QSlider(Qt.Horizontal)
        self._threshold_slider.setRange(0, 100)
        self._threshold_slider.setValue(50)
        self._threshold_slider.setTickPosition(QSlider.TicksBelow)
        self._threshold_slider.setTickInterval(10)
        self._threshold_slider.valueChanged.connect(self._on_threshold_slider)
        thresh_layout.addWidget(self._threshold_slider)

        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setValue(0.5)
        self._threshold_spin.valueChanged.connect(self._on_threshold_spin)
        thresh_layout.addWidget(self._threshold_spin)
        layout.addLayout(thresh_layout)

        # Candidate list
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._stats_label)

        self._candidate_list = QListWidget()
        self._candidate_list.itemChanged.connect(self._on_item_changed)
        self._candidate_list.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self._candidate_list)

        # Action buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.clicked.connect(self._deselect_all)
        btn_layout.addWidget(deselect_all_btn)

        confirm_btn = QPushButton("确认标注")
        confirm_btn.clicked.connect(self._confirm)
        confirm_btn.setStyleSheet(
            "background: #4ECDC4; color: #1e1e1e; font-weight: bold; padding: 4px 12px;"
        )
        btn_layout.addWidget(confirm_btn)
        layout.addLayout(btn_layout)

        self.setMinimumWidth(220)

    def set_candidates(self, candidates: List[dict]):
        """Load candidates from SAM3 predictor results."""
        self._candidates.clear()
        self._candidate_list.blockSignals(True)
        self._candidate_list.clear()

        threshold = self._threshold_spin.value()
        shown = 0
        for i, c in enumerate(candidates):
            if c["score"] < threshold:
                continue
            item = CandidateItem(c["mask"], c["score"], c["bbox"], i)
            self._candidates.append(item)

            text = f"#{i + 1}  score: {c['score']:.3f}  area: {c['mask'].sum()}"
            list_item = QListWidgetItem(text)
            list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
            list_item.setCheckState(Qt.Checked if item.checked else Qt.Unchecked)
            list_item.setData(Qt.UserRole, i)
            self._candidate_list.addItem(list_item)
            shown += 1

        self._candidate_list.blockSignals(False)
        self._stats_label.setText(f"共 {len(candidates)} 个候选，显示 {shown} 个")

    def get_checked_candidates(self) -> List[CandidateItem]:
        return [c for c in self._candidates if c.checked]

    # ── slots ─────────────────────────────────────────────────

    def _on_search(self):
        text = self._text_input.text().strip()
        if text:
            self.text_search_requested.emit(text)

    def _on_threshold_slider(self, value: int):
        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(value / 100.0)
        self._threshold_spin.blockSignals(False)
        self.threshold_changed.emit(value / 100.0)

    def _on_threshold_spin(self, value: float):
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(int(value * 100))
        self._threshold_slider.blockSignals(False)
        self.threshold_changed.emit(value)

    def _on_item_changed(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        for c in self._candidates:
            if c.index == idx:
                c.checked = item.checkState() == Qt.Checked
                break

    def _on_item_clicked(self, item: QListWidgetItem):
        idx = item.data(Qt.UserRole)
        self.candidate_hovered.emit(idx)

    def _select_all(self):
        self._candidate_list.blockSignals(True)
        for i in range(self._candidate_list.count()):
            item = self._candidate_list.item(i)
            item.setCheckState(Qt.Checked)
        for c in self._candidates:
            c.checked = True
        self._candidate_list.blockSignals(False)

    def _deselect_all(self):
        self._candidate_list.blockSignals(True)
        for i in range(self._candidate_list.count()):
            item = self._candidate_list.item(i)
            item.setCheckState(Qt.Unchecked)
        for c in self._candidates:
            c.checked = False
        self._candidate_list.blockSignals(False)

    def _confirm(self):
        checked = self.get_checked_candidates()
        if not checked:
            return
        self.candidates_confirmed.emit(checked)

    def clear(self):
        self._candidates.clear()
        self._candidate_list.clear()
        self._stats_label.setText("")
