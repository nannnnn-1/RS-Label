"""Integration tests for core data model and I/O."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core.shape import Shape, ShapeType, get_label_color
from src.core.label_data import LabelData
from src.core.io_manager import IOManager
from src.core.mask_utils import mask_to_polygons, largest_polygon


def test_shape_roundtrip():
    """Shape dict serialization is labelme-compatible."""
    shape = Shape(
        label="cat",
        points=[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        shape_type=ShapeType.POLYGON,
        group_id=None,
    )
    d = shape.to_dict()
    restored = Shape.from_dict(d)
    assert restored.label == "cat"
    assert restored.points == [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
    assert restored.shape_type == ShapeType.POLYGON
    assert restored.group_id is None


def test_rectangle_shape():
    """Rectangle shape serialization."""
    shape = Shape(
        label="car",
        points=[[5, 10], [100, 200]],
        shape_type=ShapeType.RECTANGLE,
    )
    d = shape.to_dict()
    assert d["shape_type"] == "rectangle"
    restored = Shape.from_dict(d)
    assert restored.shape_type == ShapeType.RECTANGLE


def test_label_data_roundtrip():
    """Full LabelData roundtrip through JSON."""
    ld = LabelData(
        version="5.0.0",
        image_path="/tmp/test.jpg",
        image_height=480,
        image_width=640,
        shapes=[
            Shape("cat", [[0, 0], [10, 0], [10, 10]]),
            Shape("dog", [[20, 20], [30, 20], [30, 30]], ShapeType.RECTANGLE),
        ],
    )
    data = ld.to_dict()

    assert data["version"] == "5.0.0"
    assert len(data["shapes"]) == 2
    assert data["imageHeight"] == 480
    assert data["imageWidth"] == 640
    assert data["imageData"] is None  # saved without image data

    # Restore
    restored = LabelData.from_dict(data)
    assert len(restored.shapes) == 2
    assert restored.shapes[0].label == "cat"
    assert restored.shapes[1].shape_type == ShapeType.RECTANGLE


def test_labelme_compatibility():
    """Verify we can parse real labelme JSON."""
    sample = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "person",
                "points": [[100, 200], [300, 200], [300, 400], [100, 400]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": "sample.jpg",
        "imageData": None,
        "imageHeight": 600,
        "imageWidth": 800,
    }
    ld = LabelData.from_dict(sample)
    assert ld.version == "5.0.0"
    assert ld.shapes[0].label == "person"
    assert ld.shapes[0].shape_type == ShapeType.POLYGON
    assert ld.image_path == "sample.jpg"

    # Write and read back
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(ld.to_dict(), f)
        path = f.name
    try:
        ld2 = IOManager.load_label_file(path)
        assert len(ld2.shapes) == 1
        assert ld2.shapes[0].label == "person"
    finally:
        os.unlink(path)


def test_get_label_color():
    """Label colors are consistent."""
    c1 = get_label_color("cat", 0)
    c2 = get_label_color("cat", 0)
    assert c1 == c2  # same label+index = same color
    c3 = get_label_color("dog", 1)
    assert c1 != c3  # different index = different color


def test_mask_to_polygons():
    """Mask to polygon conversion."""
    # Create a simple square mask
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    polygons = mask_to_polygons(mask, epsilon_factor=0.01)
    assert len(polygons) == 1
    assert len(polygons[0]) >= 4  # at least 4 vertices


def test_largest_polygon():
    """largest_polygon returns the biggest contour."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:30, 10:30] = 255  # small square
    mask[40:90, 40:90] = 255  # larger square

    poly = largest_polygon(mask)
    assert len(poly) >= 4

    # Verify it's the larger polygon
    pts = np.array(poly)
    x_range = pts[:, 0].max() - pts[:, 0].min()
    assert x_range > 30  # larger than the small square


if __name__ == "__main__":
    test_shape_roundtrip()
    print("PASS: test_shape_roundtrip")
    test_rectangle_shape()
    print("PASS: test_rectangle_shape")
    test_label_data_roundtrip()
    print("PASS: test_label_data_roundtrip")
    test_labelme_compatibility()
    print("PASS: test_labelme_compatibility")
    test_get_label_color()
    print("PASS: test_get_label_color")
    test_mask_to_polygons()
    print("PASS: test_mask_to_polygons")
    test_largest_polygon()
    print("PASS: test_largest_polygon")
    print("\nAll core tests passed!")
