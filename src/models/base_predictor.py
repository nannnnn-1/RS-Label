from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np


class BasePredictor(ABC):
    @abstractmethod
    def is_loaded(self) -> bool: ...

    def is_ready(self) -> bool:
        """Model loaded AND image embedding computed."""
        return self.is_loaded()

    @abstractmethod
    def load_model(self, model_name: str) -> str: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    @abstractmethod
    def set_image(self, image: np.ndarray) -> None: ...

    @abstractmethod
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def available_models(self) -> List[str]: ...

    @abstractmethod
    def current_model_name(self) -> str: ...

    @abstractmethod
    def device_name(self) -> str: ...
