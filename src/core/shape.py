from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from PySide6.QtGui import QColor

LABEL_COLORS = [
    QColor("#FF6B6B"), QColor("#4ECDC4"), QColor("#45B7D1"),
    QColor("#96CEB4"), QColor("#FFEAA7"), QColor("#DDA0DD"),
    QColor("#98D8C8"), QColor("#F7DC6F"), QColor("#BB8FCE"),
    QColor("#85C1E9"), QColor("#F8B500"), QColor("#00CED1"),
    QColor("#FF8C42"), QColor("#9B59B6"), QColor("#2ECC71"),
    QColor("#E74C3C"), QColor("#3498DB"), QColor("#1ABC9C"),
    QColor("#F39C12"), QColor("#8E44AD"),
]


def get_label_color(label: str, index: int) -> QColor:
    return LABEL_COLORS[index % len(LABEL_COLORS)]


class ShapeType(Enum):
    POLYGON = "polygon"
    RECTANGLE = "rectangle"


@dataclass
class Shape:
    label: str
    points: List[List[float]]
    shape_type: ShapeType = ShapeType.POLYGON
    group_id: Optional[int] = None
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "points": self.points,
            "shape_type": self.shape_type.value,
            "group_id": self.group_id,
            "flags": self.flags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Shape":
        return cls(
            label=d["label"],
            points=d["points"],
            shape_type=ShapeType(d.get("shape_type", "polygon")),
            group_id=d.get("group_id"),
            flags=d.get("flags", {}),
        )

    def clone(self) -> "Shape":
        return Shape(
            label=self.label,
            points=[p[:] for p in self.points],
            shape_type=self.shape_type,
            group_id=self.group_id,
            flags=dict(self.flags),
        )
