from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QLineF,
    QPointF,
    QRectF,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QCursor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QMenu,
    QWidget,
)

from ..core.shape import Shape, ShapeType, get_label_color

VERTEX_RADIUS = 4


class ToolType(Enum):
    SELECT = auto()
    POLYGON = auto()
    RECTANGLE = auto()


class CanvasMode(Enum):
    MANUAL = auto()
    SAM2 = auto()
    SAM3 = auto()


class AnnotationTool:
    """Base class for annotation tools."""

    def __init__(self, canvas: "AnnotationCanvas"):
        self.canvas = canvas

    def mouse_press(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        return False

    def mouse_move(self, event: QMouseEvent, scene_pos: QPointF):
        pass

    def mouse_release(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        return False

    def mouse_double_click(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        return False

    def key_press(self, event: QKeyEvent) -> bool:
        return False

    def activate(self):
        pass

    def deactivate(self):
        pass


class PolygonTool(AnnotationTool):
    """Click to add vertices. Click near first point, right-click, or
    double-click to close. Labelme-style snap-to-first-point."""

    SNAP_RADIUS = 10

    def __init__(self, canvas: "AnnotationCanvas"):
        super().__init__(canvas)
        self._points: List[QPointF] = []
        self._preview: Optional[QGraphicsPathItem] = None
        self._vertex_items: List[QGraphicsEllipseItem] = []
        self._snap_highlight: Optional[QGraphicsEllipseItem] = None
        self._is_snapping: bool = False

    def _dist_to_first(self, pos: QPointF) -> float:
        if not self._points:
            return float("inf")
        p0 = self._points[0]
        return ((pos.x() - p0.x()) ** 2 + (pos.y() - p0.y()) ** 2) ** 0.5

    def mouse_press(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        if event.button() == Qt.LeftButton:
            if len(self._points) >= 3 and self._is_snapping:
                # Click near first point → close polygon
                self._finish()
                return True
            self._points.append(scene_pos)
            self._update_preview()
            return True
        elif event.button() == Qt.RightButton and len(self._points) >= 3:
            self._finish()
            return True
        return False

    def mouse_double_click(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        if len(self._points) >= 3:
            # Remove the point added by the first click of the double-click
            if self._points:
                self._points.pop()
            self._finish()
            return True
        return False

    def mouse_move(self, event: QMouseEvent, scene_pos: QPointF):
        if len(self._points) >= 3 and self._dist_to_first(scene_pos) < self.SNAP_RADIUS:
            self._is_snapping = True
            self._update_preview(cursor_pos=None)  # don't draw cursor line when snapping
        else:
            self._is_snapping = False
            self._update_preview(cursor_pos=scene_pos)

    def key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Escape:
            self._clear()
            return True
        return False

    def _update_preview(self, cursor_pos: QPointF = None):
        if self._preview:
            self.canvas.scene().removeItem(self._preview)
        for v in self._vertex_items:
            self.canvas.scene().removeItem(v)
        self._vertex_items.clear()
        if self._snap_highlight:
            self.canvas.scene().removeItem(self._snap_highlight)
            self._snap_highlight = None

        if not self._points:
            return

        # Draw all vertices
        for i, pt in enumerate(self._points):
            is_first = (i == 0)
            if is_first and self._is_snapping:
                # Highlight first point when snapping: larger, filled ring
                r = self.SNAP_RADIUS
                color = QColor(0, 255, 100, 220)
                self._snap_highlight = self.canvas.scene().addEllipse(
                    pt.x() - r, pt.y() - r, r * 2, r * 2,
                    QPen(color, 2), QBrush(color)
                )
                self._snap_highlight.setZValue(35)
                # Also draw the normal vertex dot on top
                sr = VERTEX_RADIUS
                item = self.canvas.scene().addEllipse(
                    pt.x() - sr, pt.y() - sr, sr * 2, sr * 2,
                    QPen(Qt.green, 1), QBrush(Qt.green)
                )
                self._vertex_items.append(item)
            else:
                r = VERTEX_RADIUS
                item = self.canvas.scene().addEllipse(
                    pt.x() - r, pt.y() - r, r * 2, r * 2,
                    QPen(Qt.green, 1), QBrush(Qt.green)
                )
                self._vertex_items.append(item)

        # Draw lines between vertices
        path = QPainterPath()
        path.moveTo(self._points[0])
        for pt in self._points[1:]:
            path.lineTo(pt)
        if cursor_pos and not self._is_snapping:
            path.lineTo(cursor_pos)

        pen = QPen(QColor(0, 255, 0, 180), 2)
        pen.setStyle(Qt.DashLine)
        self._preview = self.canvas.scene().addPath(path, pen)

    def _finish(self):
        # Save points BEFORE _clear() which empties the list
        saved = [QPointF(p) for p in self._points]
        self._clear()
        if len(saved) >= 3:
            pts = [[p.x(), p.y()] for p in saved]
            label = self.canvas._current_label
            shape = Shape(label=label, points=pts, shape_type=ShapeType.POLYGON)
            self.canvas._add_shape(shape)
            self.canvas._suppress_context_menu = True

    def _clear(self):
        if self._preview:
            self.canvas.scene().removeItem(self._preview)
            self._preview = None
        for v in self._vertex_items:
            self.canvas.scene().removeItem(v)
        self._vertex_items.clear()
        if self._snap_highlight:
            self.canvas.scene().removeItem(self._snap_highlight)
            self._snap_highlight = None
        self._points.clear()
        self._is_snapping = False

    def deactivate(self):
        self._clear()


class RectangleTool(AnnotationTool):
    """Click and drag to draw a rectangle."""

    def __init__(self, canvas: "AnnotationCanvas"):
        super().__init__(canvas)
        self._start: Optional[QPointF] = None
        self._preview: Optional[QGraphicsPathItem] = None

    def mouse_press(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        if event.button() == Qt.LeftButton:
            self._start = scene_pos
            return True
        return False

    def mouse_move(self, event: QMouseEvent, scene_pos: QPointF):
        if self._start is None:
            return
        if self._preview:
            self.canvas.scene().removeItem(self._preview)

        rect = QRectF(self._start, scene_pos).normalized()
        path = QPainterPath()
        path.addRect(rect)
        pen = QPen(QColor(0, 255, 0, 180), 2)
        pen.setStyle(Qt.DashLine)
        self._preview = self.canvas.scene().addPath(path, pen)

    def mouse_release(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        if self._start is None:
            return False
        if self._preview:
            self.canvas.scene().removeItem(self._preview)
            self._preview = None

        rect = QRectF(self._start, scene_pos).normalized()
        if rect.width() > 3 and rect.height() > 3:
            pts = [
                [rect.left(), rect.top()],
                [rect.right(), rect.bottom()],
            ]
            label = self.canvas._current_label
            shape = Shape(label=label, points=pts, shape_type=ShapeType.RECTANGLE)
            self.canvas._add_shape(shape)

        self._start = None
        return True

    def key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Escape:
            self._start = None
            if self._preview:
                self.canvas.scene().removeItem(self._preview)
                self._preview = None
            return True
        return False

    def deactivate(self):
        self._start = None
        if self._preview:
            self.canvas.scene().removeItem(self._preview)
            self._preview = None


class SelectTool(AnnotationTool):
    """Select, move, and edit shapes."""

    def __init__(self, canvas: "AnnotationCanvas"):
        super().__init__(canvas)
        self._drag_mode: Optional[str] = None  # "vertex", "body", None
        self._drag_vertex_idx: int = -1
        self._drag_start: QPointF = QPointF()
        self._drag_shape_original: List[List[float]] = []

    def _find_shape_and_vertex(self, scene_pos: QPointF, radius: float = 8):
        """Find shape and vertex near position."""
        pos = (scene_pos.x(), scene_pos.y())
        # Check vertices first
        for shape_idx, shape in enumerate(self.canvas._data.shapes):
            for vi, pt in enumerate(shape.points):
                dx = pt[0] - pos[0]
                dy = pt[1] - pos[1]
                if (dx * dx + dy * dy) < radius * radius:
                    return shape_idx, vi
        # Check shape bodies
        for shape_idx, shape in enumerate(self.canvas._data.shapes):
            if self._hit_test_shape(shape, pos):
                return shape_idx, -1
        return -1, -1

    def _hit_test_shape(self, shape: Shape, pos: Tuple[float, float]) -> bool:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QPolygonF

        poly = QPolygonF([QPointF(p[0], p[1]) for p in shape.points])
        return poly.containsPoint(QPointF(*pos), Qt.OddEvenFill)

    def mouse_press(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        if event.button() != Qt.LeftButton:
            return False

        shape_idx, vertex_idx = self._find_shape_and_vertex(scene_pos)
        if shape_idx >= 0:
            self.canvas._selected_shape_idx = shape_idx
            self.canvas._update_shape_items()

            if vertex_idx >= 0:
                self._drag_mode = "vertex"
                self._drag_vertex_idx = vertex_idx
            else:
                self._drag_mode = "body"
                self._drag_shape_original = [
                    p[:] for p in self.canvas._data.shapes[shape_idx].points
                ]
            self._drag_start = scene_pos
            return True
        else:
            self.canvas._selected_shape_idx = -1
            self.canvas._update_shape_items()
            return False

    def mouse_move(self, event: QMouseEvent, scene_pos: QPointF):
        if self._drag_mode == "vertex" and self.canvas._selected_shape_idx >= 0:
            shape = self.canvas._data.shapes[self.canvas._selected_shape_idx]
            shape.points[self._drag_vertex_idx] = [scene_pos.x(), scene_pos.y()]
            self.canvas._update_shape_items()
            self.canvas.shape_edited.emit(self.canvas._selected_shape_idx)
        elif self._drag_mode == "body" and self.canvas._selected_shape_idx >= 0:
            dx = scene_pos.x() - self._drag_start.x()
            dy = scene_pos.y() - self._drag_start.y()
            shape = self.canvas._data.shapes[self.canvas._selected_shape_idx]
            for i, pt in enumerate(self._drag_shape_original):
                shape.points[i] = [pt[0] + dx, pt[1] + dy]
            self.canvas._update_shape_items()
            self.canvas.shape_edited.emit(self.canvas._selected_shape_idx)

    def mouse_release(self, event: QMouseEvent, scene_pos: QPointF) -> bool:
        self._drag_mode = None
        self._drag_vertex_idx = -1
        self._drag_shape_original = []
        return False

    def key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Delete and self.canvas._selected_shape_idx >= 0:
            idx = self.canvas._selected_shape_idx
            self.canvas._selected_shape_idx = -1
            del self.canvas._data.shapes[idx]
            self.canvas._update_shape_items()
            self.canvas.shape_deleted.emit(idx)
            return True
        return False

    def deactivate(self):
        self.canvas._selected_shape_idx = -1
        self.canvas._update_shape_items()
        self._drag_mode = None


class AnnotationCanvas(QGraphicsView):
    shape_added = Signal(int)       # index of new shape
    shape_edited = Signal(int)      # index of edited shape
    shape_deleted = Signal(int)     # index of deleted shape
    shape_selected = Signal(int)    # index, -1 for deselect
    status_message = Signal(str)
    image_loaded = Signal(str)      # image path
    image_cleared = Signal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._image_item: Optional[QGraphicsPixmapItem] = None
        self._shape_groups: List[QGraphicsItemGroup] = []
        self._data = None  # LabelData, set when image loaded
        self._current_label: str = ""

        self._selected_shape_idx: int = -1
        self._suppress_context_menu: bool = False

        self._pixmap: Optional[QPixmap] = None
        self._image_path: str = ""

        # Tools
        self._tool_type = ToolType.POLYGON
        self._mode = CanvasMode.MANUAL
        self._tools = {
            ToolType.SELECT: SelectTool(self),
            ToolType.POLYGON: PolygonTool(self),
            ToolType.RECTANGLE: RectangleTool(self),
        }
        self._active_tool: AnnotationTool = self._tools[ToolType.POLYGON]

        # SAM2 predictor (set by main window)
        self._sam2_predictor = None
        self._sam2_tool: Optional["Sam2Tool"] = None
        self._numpy_image: Optional[np.ndarray] = None

        # SAM3 preview items
        self._sam3_mask_items: List[QGraphicsPixmapItem] = []
        self._sam3_overlay_refs: List[np.ndarray] = []

        self._setup_view()

    def _setup_view(self):
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Background
        self.setBackgroundBrush(QBrush(QColor("#2d2d2d")))

        # Context menu for setting label
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    # ── Mode management ───────────────────────────────────────

    def set_mode(self, mode: CanvasMode):
        if mode == self._mode:
            return
        self._active_tool.deactivate()
        self._mode = mode
        if mode == CanvasMode.MANUAL:
            self._switch_to_manual()
        elif mode == CanvasMode.SAM2:
            self._switch_to_sam2()

    def current_mode(self) -> CanvasMode:
        return self._mode

    def _switch_to_manual(self):
        self._active_tool = self._tools[self._tool_type]
        self._active_tool.activate()
        self._update_cursor()

    def _switch_to_sam2(self):
        if self._sam2_tool is None:
            from .sam2_tool import Sam2Tool
            self._sam2_tool = Sam2Tool(self)
        self._active_tool = self._sam2_tool
        self._active_tool.activate()
        self.setCursor(Qt.CrossCursor)
        # Pre-compute embedding if image is loaded and predictor is ready
        self._update_sam2_embedding()

    def set_sam2_predictor(self, predictor):
        self._sam2_predictor = predictor
        # If currently in SAM2 mode, try to compute embedding
        if self._mode == CanvasMode.SAM2:
            self._update_sam2_embedding()

    def sam2_predictor(self):
        return self._sam2_predictor

    def _update_sam2_embedding(self):
        if self._sam2_predictor is None or not self._sam2_predictor.is_loaded():
            return
        if self._numpy_image is None:
            return
        try:
            self._sam2_predictor.set_image(self._numpy_image)
            self.status_message.emit("SAM2 embedding 已就绪")
        except Exception as e:
            self.status_message.emit(f"SAM2 embedding 失败: {e}")

    # ── Tool management ───────────────────────────────────────

    def set_tool(self, tool_type: ToolType):
        if tool_type not in self._tools:
            return
        self._active_tool.deactivate()
        self._tool_type = tool_type
        self._active_tool = self._tools[tool_type]
        self._active_tool.activate()
        self._update_cursor()

    def set_current_label(self, label: str):
        self._current_label = label

    def current_label(self) -> str:
        return self._current_label

    @property
    def active_tool_type(self) -> ToolType:
        return self._tool_type

    def load_image(self, image_path: str, image: np.ndarray = None):
        self.clear_image()

        if image is None:
            from ..core.io_manager import IOManager
            image = IOManager.load_image(image_path)

        h, w, _ = image.shape
        self._numpy_image = image.copy()  # keep for SAM
        self._pixmap = self._ndarray_to_pixmap(image)
        self._image_item = self._scene.addPixmap(self._pixmap)
        self._image_item.setZValue(0)
        self._scene.setSceneRect(0, 0, w, h)
        self._image_path = image_path

        from ..core.label_data import LabelData
        self._data = LabelData(
            image_path=image_path,
            image_height=h,
            image_width=w,
        )

        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self.image_loaded.emit(image_path)
        self.status_message.emit(f"已加载: {image_path} ({w}x{h})")

        # Pre-compute SAM2 embedding if in SAM2 mode
        if self._mode == CanvasMode.SAM2:
            self._update_sam2_embedding()

    def clear_image(self):
        self._active_tool.deactivate()
        self._clear_sam3_previews()
        self._scene.clear()
        self._shape_groups.clear()
        self._image_item = None
        self._pixmap = None
        self._numpy_image = None
        self._data = None
        self._selected_shape_idx = -1
        self._image_path = ""
        self.image_cleared.emit()

    def set_label_data(self, label_data):
        """Load shapes from LabelData into the canvas."""
        self._data = label_data
        self._update_shape_items()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def get_label_data(self):
        return self._data

    def has_image(self) -> bool:
        return self._image_item is not None

    def image_path(self) -> str:
        return self._image_path

    def selected_shape_index(self) -> int:
        return self._selected_shape_idx

    # ── SAM3 preview ──────────────────────────────────────────

    def show_sam3_mask_preview(self, masks: list, scores: list = None):
        """Show SAM3 candidate mask overlays on the canvas.
        masks: list of (np.ndarray, score) tuples or list of np.ndarray"""
        self._clear_sam3_previews()
        if not masks:
            return

        for i, entry in enumerate(masks):
            if isinstance(entry, tuple):
                mask, score = entry
            else:
                mask = entry
                score = None

            h, w = mask.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            color = QColor(100, 220, 220, 80)  # teal semi-transparent
            overlay[mask > 0] = [color.red(), color.green(), color.blue(), color.alpha()]
            overlay = np.ascontiguousarray(overlay)

            qimg = QImage(overlay.data, w, h, w * 4, QImage.Format_ARGB32)
            item = self._scene.addPixmap(QPixmap.fromImage(qimg))
            item.setZValue(24 + i)
            item.setVisible(True)
            self._sam3_mask_items.append(item)
            self._sam3_overlay_refs.append(overlay)

    def highlight_sam3_candidate(self, index: int):
        """Highlight a specific SAM3 candidate mask."""
        for i, item in enumerate(self._sam3_mask_items):
            if i == index:
                item.setOpacity(1.0)
                item.setZValue(26)
            else:
                item.setOpacity(0.3)
                item.setZValue(24)

    def clear_sam3_previews(self):
        self._clear_sam3_previews()

    def _clear_sam3_previews(self):
        for item in self._sam3_mask_items:
            self._scene.removeItem(item)
        self._sam3_mask_items.clear()
        self._sam3_overlay_refs.clear()

    def _add_shape(self, shape: Shape):
        if self._data is None:
            return
        self._data.shapes.append(shape)
        self._update_shape_items()
        idx = len(self._data.shapes) - 1
        self.shape_added.emit(idx)
        self.status_message.emit(f"已添加: {shape.label} ({shape.shape_type.value})")

    def _update_shape_items(self):
        for group in self._shape_groups:
            self._scene.removeItem(group)
        self._shape_groups.clear()

        if self._data is None:
            return

        for idx, shape in enumerate(self._data.shapes):
            group = self._create_shape_group(shape, idx)
            self._shape_groups.append(group)

    def _create_shape_group(self, shape: Shape, idx: int) -> QGraphicsItemGroup:
        group = QGraphicsItemGroup()
        group.setAcceptedMouseButtons(Qt.NoButton)
        group.setAcceptHoverEvents(False)
        group.setZValue(10)

        color = get_label_color(shape.label, idx)
        selected = idx == self._selected_shape_idx
        alpha = 50 if selected else 30
        line_width = 3 if selected else 2

        fill_color = QColor(color.red(), color.green(), color.blue(), alpha)
        line_color = QColor(
            color.red(), color.green(), color.blue(), 220
        )

        if shape.shape_type == ShapeType.RECTANGLE and len(shape.points) == 2:
            x0, y0 = shape.points[0]
            x1, y1 = shape.points[1]
            rect = QRectF(
                min(x0, x1), min(y0, y1),
                abs(x1 - x0), abs(y1 - y0)
            )
            poly_item = QGraphicsPolygonItem()
            poly_item.setPolygon(QPolygonF([
                QPointF(rect.left(), rect.top()),
                QPointF(rect.right(), rect.top()),
                QPointF(rect.right(), rect.bottom()),
                QPointF(rect.left(), rect.bottom()),
            ]))
        else:
            poly = QPolygonF([QPointF(p[0], p[1]) for p in shape.points])
            poly_item = QGraphicsPolygonItem()
            poly_item.setPolygon(poly)

        poly_item.setBrush(QBrush(fill_color))
        poly_item.setPen(QPen(line_color, line_width))
        group.addToGroup(poly_item)

        # Vertex handles (only when selected)
        if selected:
            r = VERTEX_RADIUS
            for pt in shape.points:
                handle = QGraphicsEllipseItem(
                    pt[0] - r, pt[1] - r, r * 2, r * 2
                )
                handle.setPen(QPen(Qt.white, 1))
                handle.setBrush(QBrush(color))
                handle.setZValue(20)
                group.addToGroup(handle)

        # Label text
        if shape.points:
            cx = sum(p[0] for p in shape.points) / len(shape.points)
            cy = min(p[1] for p in shape.points) - 8

            display_label = shape.label or "(未命名)"
            text_item = QGraphicsSimpleTextItem(display_label)
            text_item.setPos(cx, cy)
            text_item.setFont(QFont("Arial", 10, QFont.Bold))
            text_item.setBrush(QBrush(Qt.white))
            text_item.setPen(QPen(Qt.black, 1))

            # Center text
            text_rect = text_item.boundingRect()
            text_item.setPos(cx - text_rect.width() / 2, cy)
            text_item.setZValue(20)
            group.addToGroup(text_item)

        self._scene.addItem(group)
        return group

    def _on_context_menu(self, pos):
        """Context menu for shapes - assign label."""
        if self._suppress_context_menu:
            self._suppress_context_menu = False
            return
        if not self._data or not self._data.shapes:
            return

        scene_pos = self.mapToScene(pos)
        select_tool = self._tools[ToolType.SELECT]
        shape_idx, _ = select_tool._find_shape_and_vertex(scene_pos, radius=20)
        if shape_idx < 0:
            return

        self._selected_shape_idx = shape_idx
        self._update_shape_items()

        menu = QMenu(self)
        menu.addAction(f"当前标签: {self._data.shapes[shape_idx].label}")
        menu.addSeparator()
        menu.addAction("删除此标注 (Delete)", lambda: self._delete_selected())
        menu.exec(self.mapToGlobal(pos))

    def _delete_selected(self):
        if self._selected_shape_idx >= 0:
            idx = self._selected_shape_idx
            self._selected_shape_idx = -1
            del self._data.shapes[idx]
            self._update_shape_items()
            self.shape_deleted.emit(idx)

    def _ndarray_to_pixmap(self, img: np.ndarray) -> QPixmap:
        h, w, c = img.shape
        from PySide6.QtGui import QImage
        qimg = QImage(img.data, w, h, c * w, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())

    def _update_cursor(self):
        if self._tool_type == ToolType.POLYGON:
            self.setCursor(Qt.CrossCursor)
        elif self._tool_type == ToolType.RECTANGLE:
            self.setCursor(Qt.CrossCursor)
        elif self._tool_type == ToolType.SELECT:
            self.setCursor(Qt.ArrowCursor)

    # --- Zoom ---
    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)

    # --- Mouse events ---
    def mousePressEvent(self, event: QMouseEvent):
        if not self.has_image():
            super().mousePressEvent(event)
            return

        # Middle button for pan
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake_event = QMouseEvent(
                event.type(), event.position(), Qt.LeftButton,
                Qt.LeftButton, event.modifiers()
            )
            super().mousePressEvent(fake_event)
            return

        scene_pos = self.mapToScene(event.pos())
        handled = self._active_tool.mouse_press(event, scene_pos)
        if not handled:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.has_image():
            super().mouseMoveEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        self._active_tool.mouse_move(event, scene_pos)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.has_image():
            super().mouseReleaseEvent(event)
            return

        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            return

        scene_pos = self.mapToScene(event.pos())
        handled = self._active_tool.mouse_release(event, scene_pos)
        if not handled:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if not self.has_image():
            super().mouseDoubleClickEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())
        handled = self._active_tool.mouse_double_click(event, scene_pos)
        if not handled:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if not self.has_image():
            super().keyPressEvent(event)
            return

        # Delete key works globally
        if event.key() == Qt.Key_Delete and self._selected_shape_idx >= 0:
            idx = self._selected_shape_idx
            self._selected_shape_idx = -1
            del self._data.shapes[idx]
            self._update_shape_items()
            self.shape_deleted.emit(idx)
            return

        handled = self._active_tool.key_press(event)
        if not handled:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image_item:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
