"""Run the actual ONNX models on an image; no fake predictions or Telegram messages."""

import argparse
import json
import time
from dataclasses import asdict

import cv2

from app.core.config import settings
from app.inference.local_onnx import LocalOnnxProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to your test image")
    args = parser.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit("Image could not be decoded.")
    provider = LocalOnnxProvider(
        str(settings.resolve(settings.model_path)),
        settings.confidence_threshold,
        settings.inference_threads,
        person_model_path=str(settings.resolve(settings.person_model_path)),
    )
    provider.load()
    started = time.perf_counter()
    detections = provider.predict_sync(frame)
    print(
        json.dumps(
            {
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "detections": [asdict(d) for d in detections],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
