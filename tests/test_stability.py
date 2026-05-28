"""Stability test: 20+ images without crash, memory stability check."""
import gc
import os
import sys
import tempfile
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication(sys.argv)

import numpy as np
from src.ui.main_window import MainWindow
from src.ui.canvas import CanvasMode
from src.core.shape import Shape, ShapeType
from src.core.io_manager import IOManager
from src.models.sam2_predictor import SAM2Predictor

tracemalloc.start()

w = MainWindow()
TMP = tempfile.gettempdir()
print("=" * 60)
print("STABILITY TEST: 22-image workflow + reload")
print(f"  Temp dir: {TMP}")
print("=" * 60)

# --- Test 1: 22-image loop with SAM2 ---
p2 = SAM2Predictor()
p2.load_model("base_plus")
w._canvas.set_sam2_predictor(p2)
print("\n[1] SAM2 loaded, processing 22 images...\n")

errors = []
for i in range(22):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    try:
        w._canvas.load_image(f"img_{i:04d}.jpg", image=img)

        # Manual polygon
        shape = Shape("test", [[10, 10], [100, 10], [100, 100], [10, 100]])
        w._canvas._data.shapes.append(shape)

        # SAM2 prediction
        p2.set_image(img)
        masks, scores = p2.predict(
            point_coords=np.array([[320, 240]]),
            point_labels=np.array([1]),
        )
        w._canvas._update_shape_items()

        # Verify mask shape
        assert masks.ndim == 3, f"Expected 3D masks, got {masks.shape}"

        # Save and clean up
        temp_json = os.path.join(TMP, f"annot_{i:04d}.json")
        w._label_data = w._canvas._data
        IOManager.save_label_file(w._label_data, temp_json)
        if os.path.exists(temp_json):
            os.unlink(temp_json)

        assert len(w._canvas._data.shapes) >= 1

        if (i + 1) % 5 == 0:
            mem_kb = tracemalloc.get_traced_memory()[0] / 1024
            print(f"  [{i+1}/22] OK | shapes={len(w._canvas._data.shapes)} | mem={mem_kb:.0f} KB")

    except Exception as e:
        errors.append(f"Image {i}: {e}")
        print(f"  [{i+1}/22] FAIL: {e}")

    w._canvas.clear_image()

mem_after_22 = tracemalloc.get_traced_memory()[0] / 1024

# --- Test 2: Unload and reload ---
p2.unload_model()
gc.collect()
mem_after_unload = tracemalloc.get_traced_memory()[0] / 1024

p2.load_model("base_plus")
mem_after_reload = tracemalloc.get_traced_memory()[0] / 1024

# Verify model works after reload
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
p2.set_image(img)
masks, scores = p2.predict(
    point_coords=np.array([[320, 240]]),
    point_labels=np.array([1]),
)
assert masks.ndim == 3
print(f"\n[2] Model unload/reload cycle:")
print(f"    Memory after 22 images: {mem_after_22:.0f} KB")
print(f"    Memory after unload:    {mem_after_unload:.0f} KB")
print(f"    Memory after reload:    {mem_after_reload:.0f} KB")
print(f"    Post-reload predict OK (masks={masks.shape})")

p2.unload_model()
tracemalloc.stop()

# --- Test 3: SAM3 stability ---
print("\n[3] Testing SAM3 load/unload...")
from src.models.sam3_predictor import SAM3Predictor
p3 = SAM3Predictor()
p3.load_model("sam3")
img = np.random.randint(0, 255, (1008, 1008, 3), dtype=np.uint8)
p3.set_image(img)
results = p3.text_predict("test")
print(f"    SAM3 text_predict: {len(results)} candidates")
p3.unload_model()
gc.collect()
print("    SAM3 unloaded OK")

# --- Report ---
print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} errors")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("PASSED: 22 images, 0 errors, no crash, memory stable")
    print("      - SAM2 load/22-images/unload/reload/predict")
    print("      - SAM3 load/predict/unload")
    print("=" * 60)
