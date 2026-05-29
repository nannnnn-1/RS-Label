from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

from .base_predictor import BasePredictor

SAM2_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "third_party_repository", "sam2",
)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Models", "SAM2",
)

MODEL_VARIANTS = {
    "tiny": {
        "config": "sam2.1/sam2.1_hiera_t.yaml",
        "checkpoint": "sam2.1_hiera_tiny.pt",
    },
    "small": {
        "config": "sam2.1/sam2.1_hiera_s.yaml",
        "checkpoint": "sam2.1_hiera_small.pt",
    },
    "base_plus": {
        "config": "sam2.1/sam2.1_hiera_b+.yaml",
        "checkpoint": "sam2.1_hiera_base_plus.pt",
    },
    "large": {
        "config": "sam2.1/sam2.1_hiera_l.yaml",
        "checkpoint": "sam2.1_hiera_large.pt",
    },
}


class SAM2Predictor(BasePredictor):
    def __init__(self):
        self._model = None
        self._predictor = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._current_model_name = ""

    def is_loaded(self) -> bool:
        return self._predictor is not None

    def is_ready(self) -> bool:
        return self._predictor is not None and self._predictor._is_image_set

    def load_model(self, model_name: str) -> str:
        if model_name not in MODEL_VARIANTS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_VARIANTS.keys())}")

        variant = MODEL_VARIANTS[model_name]
        config_path = os.path.join(
            SAM2_ROOT, "sam2", "configs", variant["config"]
        )
        checkpoint_path = os.path.join(MODELS_DIR, variant["checkpoint"])

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.unload_model()

        cfg = OmegaConf.load(config_path)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
        model.load_state_dict(sd)
        model = model.to(self._device)
        model.eval()

        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self._model = model
        self._predictor = SAM2ImagePredictor(model)
        self._current_model_name = model_name

        return f"已加载 SAM2 {model_name} ({self._device})"

    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
        if self._predictor is not None:
            del self._predictor
            self._predictor = None
        self._current_model_name = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_image(self, image: np.ndarray):
        if self._predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        self._predictor.set_image(image)

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (masks, scores) as numpy arrays."""
        if self._predictor is None:
            raise RuntimeError("Model not loaded.")

        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        return masks, scores

    def reset(self):
        if self._predictor is not None:
            self._predictor.reset_predictor()

    def available_models(self) -> List[str]:
        models = []
        for name, variant in MODEL_VARIANTS.items():
            config_path = os.path.join(SAM2_ROOT, "sam2", "configs", variant["config"])
            ckpt_path = os.path.join(MODELS_DIR, variant["checkpoint"])
            if os.path.exists(config_path) and os.path.exists(ckpt_path):
                models.append(name)
        return models

    def current_model_name(self) -> str:
        return self._current_model_name

    def device_name(self) -> str:
        return self._device
