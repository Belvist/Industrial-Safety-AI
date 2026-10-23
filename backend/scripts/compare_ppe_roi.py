"""A/B current PPE weights: full-frame vs worker-centered ROI.

Use RAW frames (without drawn detections). No mAP claims: this prints
confidence, weak flags, coordinates and actual inference latency. For ground
truth and real accuracy use training/evaluate.py with annotated footage.

From repo root with .venv already installed:
    .venv/bin/python backend/scripts/compare_ppe_roi.py /path/to/frame1.jpg /path/to/frame2.jpg
    .venv/bin/python backend/scripts/compare_ppe_roi.py data/snapshots/*.jpg --out data/ppe_ab.json
Windows: .venv\\Scripts\\python.exe backend/scripts/compare_ppe_roi.py image.jpg
"""

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
PPE_CLASSES = {"helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask"}


def run_mode(images, mode, runs):
    import cv2

    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    model = LocalOnnxProvider(
        str(settings.resolve(settings.model_path)),
        confidence=settings.confidence_threshold,
        threads=settings.inference_threads,
        person_model_path=str(settings.resolve(settings.person_model_path))
        if settings.use_separate_person_model else None,
        providers=settings.onnx_provider_list,
        thresholds=settings.class_thresholds,
        floor=settings.display_floor,
        roi_mode=mode,
    )
    if not settings.use_separate_person_model and mode == "auto":
        print("Warning: ROI mode requires USE_SEPARATE_PERSON_MODEL=true")
    results = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"Cannot read image: {path}")
        model.predict_sync(frame)  # warmup (exclude load latency)
        latencies = []
        detections = []
        for _ in range(runs):
            started = time.perf_counter()
            detections = model.predict_sync(frame)
            latencies.append((time.perf_counter() - started) * 1000)
        items = [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "weak": d.weak,
                "bbox": [round(d.bbox.x1, 4), round(d.bbox.y1, 4),
                         round(d.bbox.x2, 4), round(d.bbox.y2, 4)],
            }
            for d in detections if d.class_name in PPE_CLASSES
        ]
        results.append({
            "image": str(path), "mode": mode,
            "people": sum(1 for d in detections if d.class_name == "person" and not d.weak),
            "mean_ms": round(statistics.mean(latencies), 1),
            "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * .95) - 1)], 1),
            "ppe": items,
        })
    del model
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    for path in args.images:
        if not path.exists():
            raise SystemExit(f"Missing image: {path}")
    rows = [*run_mode(args.images, "off", args.runs),
            *run_mode(args.images, "auto", args.runs)]
    for path in args.images:
        print(f"\nIMAGE {path}:")
        for item in rows:
            if item["image"] != str(path):
                continue
            print(f"  {item['mode']} {item['mean_ms']} ms/frame "
                  f"(p95={item['p95_ms']}), persons={item['people']}")
            for det in item["ppe"]:
                state = "WEAK (not actionable)" if det["weak"] else "STRONG"
                print(f"    {det['class']} {det['confidence']:.2f} {state} bbox={det['bbox']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(f"\nSaved comparison: {args.out}")
    print("\nThese are detections, NOT accuracy metrics. Check labeled videos, "
          "worker association, false incident rates and throughput before deployment.")


if __name__ == "__main__":
    main()
