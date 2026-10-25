from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from app.domain.models import Detection

Frame = NDArray[np.uint8]


class InferenceProvider(ABC):
    """Single contract for local development and LinQ deployment."""

    @abstractmethod
    async def predict(self, frame: Frame) -> list[Detection]:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
