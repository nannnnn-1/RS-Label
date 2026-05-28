from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem,
    QGraphicsPixmapItem, QGraphicsSimpleTextItem,
)

from ..core.shape import Shape, ShapeType
from ..core.mask_utils import largest_polygon
from .canvas import AnnotationTool

POINT_RADIUS = 5
POSITIVE_COLOR = QColor(0, 255, 0, 200)
NEGATIVE_COLOR = QColor(255, 60, 60, 200)
BOX_COLOR = QColor(0, 136, 255, 180)
MASK_COLOR = QColor(0, 200, 200, 100)


class Sam2Tool(AnnotationTool):
    """Interactive SAM2 annotation using point and box prompts.

    - Left click: positive point (green)
    - Right click: negative point (red)
    - Ctrl + Left drag: bounding box (blue)
    - Enter: confirm mask -> polygon
    - Escape: clear all prompts
    - Backspace: undo last prompt
    """

    def __init__(self, canvas):
        super().__init__(canvas)
        self._positive_points: List[QPointF] = []
        self._negative_points: List[QPointF] = []
        self._box_start: Optional[QPointF] = None
        self._box_end: Optional[QPointF] = None
        self._dragging_box: bool = False

        self._point_items: List[QGraphicsEllipseItem] = []
        self._box_item: Optional[QGraphicsPathItem] = None
        self._mask_item: Optional[QGraphicsPixmapItem] = None
        self._hint_item: Optional[QGraphicsSimpleTextItem] = None

        self._current_mask: Optional[np.ndarray] = None
        self._masks_all: Optional[np.ndarray] = None
        self._scores: Optional[np.ndarray] = None

    # ── mouse events ──────────────────────────────────────────

    def mouse_press(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        if event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier:
            self._box_start = scene_pos
            self._box_end = scene_pos
            self._dragging_box = True
            return True
        elif event.button() == Qt.LeftButton:
            self._positive_points.append(scene_pos)
            self._predict()
            return True
        elif event.button() == Qt.RightButton:
            self._negative_points.append(scene_pos)
            self._predict()
            return True
        return False

    def mouse_move(self, event: QMouseEvent, scene_pos: QPointF):
        if self._dragging_box and self._box_start is not None:
            self._box_end = scene_pos
            self._draw_box_preview()

    def mouse_release(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        if self._dragging_box and self._box_start is not None:
            self._box_end = scene_pos
            self._dragging_box = False
            self._predict()
            return True
        return False

    def mouse_double_click(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        return False

    def key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Escape:
            self._clear_all()
            return True
        elif event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._confirm_mask()
            return True
        elif event.key() == Qt.Key_Backspace:
            self._undo_last()
            return True
        return False

    # ── box helpers ───────────────────────────────────────────

    def _get_box_array(self) -> Optional[np.ndarray]:
        if self._box_start is None or self._box_end is None:
            return None
        x0 = min(self._box_start.x(), self._box_end.x())
        y0 = min(self._box_start.y(), self._box_end.y())
        x1 = max(self._box_start.x(), self._box_end.x())
        y1 = max(self._box_start.y(), self._box_end.y())
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    # ── prediction ────────────────────────────────────────────

    def _predict(self):
        predictor = self.canvas.sam2_predictor()
        if predictor is None or not predictor.is_loaded():
            self.canvas.status_message.emit("模型未加载")
            return
        if not predictor.is_ready():
            self.canvas.status_message.emit("请先加载图片以计算 SAM2 embedding")
            return

        coords, labels = [], []
        for pt in self._positive_points:
            coords.append([pt.x(), pt.y()]); labels.append(1)
        for pt in self._negative_points:
            coords.append([pt.x(), pt.y()]); labels.append(0)

        pc = np.array(coords, dtype=np.float32) if coords else None
        pl = np.array(labels, dtype=np.int32) if labels else None
        box = self._get_box_array()

        if pc is None and box is None:
            self._clear_mask()
            return

        try:
            masks, scores = predictor.predict(point_coords=pc, point_labels=pl, box=box)
        except Exception as e:
            self.canvas.status_message.emit(f"预测失败: {e}")
            return

        self._masks_all = masks
        self._scores = scores
        self._current_mask = masks[int(np.argmax(scores))]
        self._draw_prompt_items()
        self._draw_mask_preview()
        self.canvas.status_message.emit(
            f"SAM2: {len(coords)} 个点, mask 置信度 {scores.max():.2f}"
        )

    # ── drawing ────────────────────────────────────────────────

    def _draw_prompt_items(self):
        for item in self._point_items:
            self.canvas.scene().removeItem(item)
        self._point_items.clear()

        r = POINT_RADIUS
        for pt in self._positive_points:
            item = self.canvas.scene().addEllipse(
                pt.x() - r, pt.y() - r, r * 2, r * 2,
                QPen(Qt.white, 2), QBrush(POSITIVE_COLOR)
            )
            item.setZValue(30)
            self._point_items.append(item)
        for pt in self._negative_points:
            item = self.canvas.scene().addEllipse(
                pt.x() - r, pt.y() - r, r * 2, r * 2,
                QPen(Qt.white, 2), QBrush(NEGATIVE_COLOR)
            )
            item.setZValue(30)
            self._point_items.append(item)

        self._draw_box_preview()

    def _draw_box_preview(self):
        if self._box_item:
            self.canvas.scene().removeItem(self._box_item)
            self._box_item = None

        box = self._get_box_array()
        if box is not None:
            pen = QPen(BOX_COLOR, 2)
            pen.setStyle(Qt.DashLine)
            rect = QRectF(float(box[0]), float(box[1]),
                           float(box[2] - box[0]), float(box[3] - box[1]))
            path = QPainterPath()
            path.addRect(rect)
            self._box_item = self.canvas.scene().addPath(path, pen)
            self._box_item.setZValue(29)
        elif self._dragging_box and self._box_start and self._box_end:
            # Draw temporary rectangle during drag
            pen = QPen(BOX_COLOR, 2)
            pen.setStyle(Qt.DashLine)
            rect = QRectF(self._box_start, self._box_end).normalized()
            path = QPainterPath()
            path.addRect(rect)
            if self._box_item:
                self.canvas.scene().removeItem(self._box_item)
            self._box_item = self.canvas.scene().addPath(path, pen)
            self._box_item.setZValue(29)

    def _draw_mask_preview(self):
        if self._mask_item:
            self.canvas.scene().removeItem(self._mask_item)
            self._mask_item = None
        if self._hint_item:
            self.canvas.scene().removeItem(self._hint_item)
            self._hint_item = None
        if self._current_mask is None:
            return

        h, w = self._current_mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        mask_px = self._current_mask > 0
        overlay[mask_px, :] = [MASK_COLOR.red(), MASK_COLOR.green(),
                               MASK_COLOR.blue(), MASK_COLOR.alpha()]
        overlay = np.ascontiguousarray(overlay)
        qimg = QImage(overlay.data, w, h, w * 4, QImage.Format_ARGB32)
        pixmap = QPixmap.fromImage(qimg)
        self._mask_item = self.canvas.scene().addPixmap(pixmap)
        self._mask_item.setZValue(25)
        self._mask_overlay_ref = overlay

        # Hint text overlay
        self._hint_item = self.canvas.scene().addSimpleText(
            "Enter 确认 | Esc 取消 | Backspace 撤销 | 左键+点 右键-点",
            QFont("Arial", 11)
        )
        self._hint_item.setBrush(QBrush(QColor(255, 255, 255, 200)))
        self._hint_item.setPen(QPen(QColor(0, 0, 0, 180), 1))
        self._hint_item.setPos(10, h - 30)
        self._hint_item.setZValue(40)
        self._hint_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

    def _clear_mask(self):
        if self._mask_item:
            self.canvas.scene().removeItem(self._mask_item)
            self._mask_item = None
        if self._hint_item:
            self.canvas.scene().removeItem(self._hint_item)
            self._hint_item = None
        self._current_mask = None
        self._masks_all = None
        self._scores = None

    # ── actions ────────────────────────────────────────────────

    def _clear_all(self):
        self._positive_points.clear()
        self._negative_points.clear()
        self._box_start = None
        self._box_end = None
        self._dragging_box = False

        for item in self._point_items:
            self.canvas.scene().removeItem(item)
        self._point_items.clear()
        if self._box_item:
            self.canvas.scene().removeItem(self._box_item)
            self._box_item = None
        self._clear_mask()

    def _undo_last(self):
        if self._box_start is not None and not self._dragging_box:
            self._box_start = None
            self._box_end = None
        elif self._negative_points:
            self._negative_points.pop()
        elif self._positive_points:
            self._positive_points.pop()

        if self._positive_points or self._negative_points or self._box_end:
            self._predict()
        else:
            self._clear_all()

    def _confirm_mask(self):
        if self._current_mask is None:
            self.canvas.status_message.emit("没有可确认的 mask")
            return

        poly = largest_polygon(self._current_mask)
        if not poly:
            self.canvas.status_message.emit("mask 轮廓提取失败")
            return

        label = self.canvas._current_label
        shape = Shape(label=label, points=poly, shape_type=ShapeType.POLYGON)
        self.canvas._add_shape(shape)
        self.canvas.status_message.emit(f"已确认: {label} (SAM2)")
        self._clear_all()

    # ── lifecycle ─────────────────────────────────────────────

    def activate(self):
        pass

    def deactivate(self):
        self._clear_all()
