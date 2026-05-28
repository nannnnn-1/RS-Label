from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .shape import Shape


@dataclass
class LabelData:
    version: str = "5.0.0"
    flags: dict = field(default_factory=dict)
    shapes: List[Shape] = field(default_factory=list)
    image_path: str = ""
    image_data: Optional[str] = None
    image_height: int = 0
    image_width: int = 0

    @property
    def file_name(self) -> str:
        import os
        return os.path.basename(self.image_path)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "flags": self.flags,
            "shapes": [s.to_dict() for s in self.shapes],
            "imagePath": self.file_name,
            "imageData": self.image_data,
            "imageHeight": self.image_height,
            "imageWidth": self.image_width,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelData":
        return cls(
            version=d.get("version", "5.0.0"),
            flags=d.get("flags", {}),
            shapes=[Shape.from_dict(s) for s in d.get("shapes", [])],
            image_path=d.get("imagePath", ""),
            image_data=d.get("imageData"),
            image_height=d.get("imageHeight", 0),
            image_width=d.get("imageWidth", 0),
        )
