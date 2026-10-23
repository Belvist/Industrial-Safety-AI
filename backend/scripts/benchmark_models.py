"""Benchmark dual-model (PPE + person) vs single-model (PPE only) inference.

Usage (from repo root, .venv active):
    python backend/scripts/benchmark_models.py [image ...] [--runs N]

Without images downloads the public Ultralytics bus.jpg sample used in
docs/VALIDATION.md. Reports honest numbers: latency/FPS and detection counts
as a *proxy* (no ground-truth labels here, so no precision/recall claims).
"""

import argparse
import statistics
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

BUS_URL = "https://github.com/ultralytics/ultralytics/raw/main/ultralytics/assets/bus.jpg"


def load_image(path: Path):
    import cv2

    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Image could not be decoded: {path}")
    return frame


def ensure_sample(path: Path) -> Path:
    if not path.exists():
        print(f"Downloading sample image {BUS_URL}", flush=True)
        urllib.request.urlretrieve(BUS_URL, path)
    return path


def bench(provider, frames, runs: int) -> dict:
    for frame in frames:
        provider.predict_sync(frame)  # warmup
    lat: list[float] = []
    counts: Counter = Counter()
    people = 0
    for _ in range(runs):
        for frame in frames:
            started = time.perf_counter()
            detections = provider.predict_sync(frame)
            lat.append((time.perf_counter() - started) * 1000)
            for det in detections:
                counts[det.class_name] += 1
                if det.class_name == "person":
                    people += 1
    runs_total = runs * len(frames)
    return {
        "mean_ms": statistics.mean(lat),
        "p95_ms": sorted(lat)[max(0, int(len(lat) * 0.95) - 1)],
        "fps": 1000 / statistics.mean(lat),
        "per_frame": {k: v / runs_total for k, v in sorted(counts.items())},
        "persons": people / runs_total,
    }


def main():
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="Test images (default: bus.jpg sample)")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    paths = [Path(p) for p in args.images] or [ensure_sample(Path("/tmp/isa-bus.jpg"))]
    frames = [load_image(p) for p in paths]
    print(f"Frames: {[str(p) for p in paths]} shapes: {[f.shape for f in frames]}")

    common = dict(
        confidence=settings.confidence_threshold,
        threads=settings.inference_threads,
        thresholds=settings.class_thresholds,
    )
    dual = LocalOnnxProvider(
        str(settings.resolve(settings.model_path)),
        person_model_path=str(settings.resolve(settings.person_model_path)),
        **common,
    )
    single = LocalOnnxProvider(str(settings.resolve(settings.model_path)), **common)
    results = {"A dual (PPE+person)": bench(dual, frames, args.runs)}
    results["B single (PPE only)"] = bench(single, frames, args.runs)
    for name, res in results.items():
        print(f"\n{name}: {res['mean_ms']:.0f} ms/frame (p95 {res['p95_ms']:.0f}), "
              f"{res['fps']:.1f} FPS")
        print(f"  detections/frame: {res['per_frame']}")
    a, b = results["A dual (PPE+person)"], results["B single (PPE only)"]
    print(f"\nSingle-model is {a['mean_ms'] / b['mean_ms']:.2f}x the dual latency; "
          f"person proxy {a['persons']:.2f} vs {b['persons']:.2f} per frame.")


if __name__ == "__main__":
    main()
