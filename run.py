#!/usr/bin/env python
"""Quick launch script for SAM Annotation Tool."""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Add third-party repos for model loading
THIRD_PARTY = os.path.join(PROJECT_ROOT, "third_party_repository")
SAM2_REPO = os.path.join(THIRD_PARTY, "sam2")
SAM3_REPO = os.path.join(THIRD_PARTY, "sam3")
for p in [SAM2_REPO, SAM3_REPO]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

if __name__ == "__main__":
    print("启动 SAM 数据标注工具...")
    print("  环境: data_annotation (conda)")
    print("  模式一: 手动标注 (多边形/矩形)")

    app = QApplication(sys.argv)
    app.setApplicationName("SAM 数据标注工具")

    from src.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    print("  窗口已启动，请打开图片目录或图片文件开始标注。")
    sys.exit(app.exec())
