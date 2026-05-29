from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np
import torch

from .base_predictor import BasePredictor

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SAM3_ROOT = os.path.join(PROJECT_ROOT, "third_party_repository", "sam3")
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models", "SAM3")

if SAM3_ROOT not in sys.path:
    sys.path.insert(0, SAM3_ROOT)


class SAM3Predictor(BasePredictor):
    """SAM3 text-prompt + geometric-prompt grounding."""

    MODEL_VARIANTS = {
        "sam3": "sam3.pt",
        "sam3.1": "sam3.1_multiplex.pt",
    }

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._current_model_name = ""
        self._state = {}

    def is_loaded(self) -> bool:
        return self._model is not None

    def _patch_dtype(self):
        """Patch SAM3's fused ops to preserve consistent dtype.
        The original addmm_act converts to bfloat16 internally, which
        breaks when model params are stored as float32."""
        import sam3.perflib.fused as fused_mod

        _original = fused_mod.addmm_act

        def patched(activation, linear, mat1):
            target_dtype = linear.weight.dtype
            out = _original(activation, linear, mat1)
            if out.dtype != target_dtype:
                out = out.to(target_dtype)
            return out

        fused_mod.addmm_act = patched
        # Patch vitdet reference (loaded by model_builder import chain)
        import sam3.model.vitdet as vitdet_mod
        vitdet_mod.addmm_act = patched

    def load_model(self, model_name: str) -> str:
        if model_name not in self.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown model: {model_name}. Available: {list(self.MODEL_VARIANTS.keys())}"
            )

        checkpoint_path = os.path.join(MODELS_DIR, self.MODEL_VARIANTS[model_name])
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.unload_model()

        from sam3.model_builder import build_sam3_image_model

        # Patch MUST happen after import (vitdet loads addmm_act) but before model build
        self._patch_dtype()

        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            device=self._device,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            load_from_HF=False,
        )

        from sam3.model.sam3_image_processor import Sam3Processor
        processor = Sam3Processor(
            model, resolution=1008, device=self._device, confidence_threshold=0.5
        )

        self._model = model
        self._processor = processor
        self._state = {}
        self._current_model_name = model_name

        return f"已加载 SAM3 {model_name} ({self._device})"

    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._state = {}
        self._current_model_name = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_image(self, image: np.ndarray):
        if self._processor is None:
            raise RuntimeError("Model not loaded.")
        import PIL.Image
        pil_img = PIL.Image.fromarray(image)
        self._state = self._processor.set_image(pil_img, self._state)

    def text_predict(self, text: str) -> List[dict]:
        """Predict masks from text prompt. Returns list of {mask, score, bbox}."""
        if self._processor is None:
            raise RuntimeError("Model not loaded.")
        if not self._state:
            raise RuntimeError("Call set_image() first.")

        self._processor.reset_all_prompts(self._state)
        self._state = self._processor.set_text_prompt(text, self._state)

        results = []
        masks = self._state.get("masks")
        scores = self._state.get("scores")
        boxes = self._state.get("boxes")

        if masks is not None:
            masks_np = masks.cpu().numpy()
            scores_np = scores.cpu().numpy()
            boxes_np = boxes.cpu().numpy()

            for i in range(masks_np.shape[0]):
                mask = masks_np[i, 0] > 0.5
                results.append({
                    "mask": mask,
                    "score": float(scores_np[i]),
                    "bbox": boxes_np[i].tolist(),
                })

        return results

    def set_confidence_threshold(self, threshold: float):
        if self._processor is not None:
            self._processor.set_confidence_threshold(threshold, self._state)

    def add_box_prompt(self, box_xyxy: list, label: bool = True):
        if self._processor is None:
            raise RuntimeError("Model not loaded.")
        h = self._state.get("original_height", 1)
        w = self._state.get("original_width", 1)
        x0, y0, x1, y1 = box_xyxy
        cx = (x0 + x1) / 2 / w
        cy = (y0 + y1) / 2 / h
        bw = (x1 - x0) / w
        bh = (y1 - y0) / h
        self._state = self._processor.add_geometric_prompt(
            [cx, cy, bw, bh], label, self._state
        )

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
    ):
        """Not used for SAM3 text mode. Use text_predict() instead."""
        raise NotImplementedError("Use text_predict() for SAM3 text mode")

    def reset(self):
        if self._processor is not None and self._state:
            self._processor.reset_all_prompts(self._state)

    def available_models(self) -> List[str]:
        models = []
        for name, file_name in self.MODEL_VARIANTS.items():
            if os.path.exists(os.path.join(MODELS_DIR, file_name)):
                models.append(name)
        return models

    def current_model_name(self) -> str:
        return self._current_model_name

    def device_name(self) -> str:
        return self._device
