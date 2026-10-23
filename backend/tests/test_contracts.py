import asyncio

import numpy as np

from app.domain.models import BoundingBox, Detection
from app.inference.base import Frame, InferenceProvider


class FakeProvider(InferenceProvider):
    @property
    def name(self) -> str:
        return "fake"

    async def predict(self, frame: Frame) -> list[Detection]:
        assert frame.shape == (32, 32, 3)
        return [
            Detection(
                class_name="person",
                confidence=0.99,
                bbox=BoundingBox(0.1, 0.1, 0.8, 0.9),
            )
        ]


def test_normalized_detection_contract() -> None:
    provider = FakeProvider()
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    detections = asyncio.run(provider.predict(frame))
    assert detections[0].class_name == "person"
    assert detections[0].confidence == 0.99
