import json
import os

import numpy as np
from PIL import Image

from .label_data import LabelData


class IOManager:
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    @staticmethod
    def load_label_file(json_path: str) -> LabelData:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        label_data = LabelData.from_dict(data)
        # Resolve image path from JSON directory + stored imagePath
        json_dir = os.path.dirname(os.path.abspath(json_path))
        img_name = data.get("imagePath", "")
        if img_name:
            candidate = os.path.join(json_dir, img_name)
            if os.path.exists(candidate):
                label_data.image_path = candidate
            else:
                label_data.image_path = json_path  # fallback
        else:
            label_data.image_path = json_path
        return label_data

    @staticmethod
    def save_label_file(label_data: LabelData, save_path: str):
        data = label_data.to_dict()
        data["imageData"] = None
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def get_label_path(image_path: str) -> str:
        base = os.path.splitext(image_path)[0]
        return base + ".json"

    @staticmethod
    def has_label_file(image_path: str) -> bool:
        return os.path.exists(IOManager.get_label_path(image_path))
