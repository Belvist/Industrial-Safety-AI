"""Compare two ONNX weights on an independent test split (YOLO txt labels).

Uses the app's own LocalOnnxProvider (real thresholds, real NMS) so numbers
transfer to production. Reports per-class P/R @ IoU 0.5, latency, and a
verdict line. Never evaluate on train/val clips.

Usage:
    python training/evaluate.py --data training/ppe-custom.yaml \\
        --base models/ppe.onnx --candidate models/ppe-custom.onnx
"""

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path


def load_metrics():
    path = Path(__file__).resolve().parent / "metrics.py"
    spec = importlib.util.spec_from_file_location("ppe_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def read_txt(path: Path, w: int, h: int):
    items = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            items.append({
                "class": cls,
                "bbox": [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
            })
    return items


def main():
    from app.inference.local_onnx import LocalOnnxProvider

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--person-model", default=None,
                        help="Dedicated person weights: compare the real dual pipeline")
    parser.add_argument("--classes", nargs="+",
                        default=["person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask"])
    parser.add_argument("--production", action="store_true",
                        help="Score with the app production per-class thresholds "
                             "(no_helmet/no_vest 0.60 etc.) instead of flat 0.4. "
                             "Weak (sub-threshold) detections are not scored, "
                             "same as violation logic.")
    args = parser.parse_args()

    import yaml

    data_path = Path(args.data).resolve()
    dataset = yaml.safe_load(data_path.read_text())
    if not isinstance(dataset, dict) or "test" not in dataset or "names" not in dataset:
        raise SystemExit("Dataset YAML must define names and independent test images.")
    names = dataset["names"]
    if isinstance(names, dict):
        if set(map(int, names)) != set(range(len(names))):
            raise SystemExit("Class indices in dataset names must be contiguous from zero.")
        classes = [str(names[i] if i in names else names[str(i)]) for i in range(len(names))]
    else:
        classes = list(names)
    if classes != args.classes:
        raise SystemExit(f"Dataset class IDs/order {classes} differ from --classes {args.classes}.")
    yaml_root = Path(dataset.get("path") or data_path.parent)
    if not yaml_root.is_absolute():
        yaml_root = (data_path.parent / yaml_root).resolve()
    test_entries = dataset["test"]
    if isinstance(test_entries, str):
        test_entries = [test_entries]
    if not isinstance(test_entries, list) or not test_entries:
        raise SystemExit("Dataset YAML test split is empty or invalid.")
    test_dirs = [(yaml_root / entry).resolve() for entry in test_entries]

    images = []
    for d in test_dirs:
        images += sorted(p for p in d.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit("No test images. Check the test: section of dataset.yaml.")
    print(f"Test images: {len(images)}")

    import cv2

    def check_mapping(weights: str) -> None:
        """Fail loudly when yaml class ids disagree with model metadata."""
        from app.inference.local_onnx import LocalOnnxProvider

        probe = LocalOnnxProvider(weights, 0.4, 2)
        probe.load()
        model_names = sorted(set(probe.classes.values()))
        yaml_names = sorted(args.classes)
        missing = [c for c in yaml_names if c not in model_names]
        if missing:
            raise SystemExit(
                f"{weights}: yaml classes {missing} absent from model heads "
                f"{model_names}. Fix the mapping, do not silently mis-score."
            )

    check_mapping(args.base)
    check_mapping(args.candidate)

    thresholds = None
    if args.production:
        from app.core.config import settings
        thresholds = settings.class_thresholds
        print(f"Production thresholds: {thresholds}")
    else:
        print("Flat detector confidence 0.4 for all classes")

    def make_provider(weights: str):
        from app.inference.local_onnx import LocalOnnxProvider
        if thresholds is None:
            return LocalOnnxProvider(
                weights, 0.4, 2,
                person_model_path=args.person_model if args.person_model else None,
            )
        return LocalOnnxProvider(
            weights, 0.4, 2,
            person_model_path=args.person_model if args.person_model else None,
            thresholds=thresholds,
        )

    for name, weights in (("base", args.base), ("candidate", args.candidate)):
        provider = make_provider(weights)
        expected_by_image: dict[str, list[tuple[str, list[float]]]] = {}
        predicted_by_image: dict[str, list[tuple[str, list[float], float]]] = {}
        lat: list[float] = []
        for path in images:
            frame = cv2.imread(str(path))
            if frame is None:
                raise SystemExit(f"Cannot decode image: {path}")
            h, w = frame.shape[:2]
            started = time.perf_counter()
            detections = provider.predict_sync(frame)
            lat.append((time.perf_counter() - started) * 1000)
            label_path = path.parent.parent / "labels" / (path.stem + ".txt")
            if not label_path.is_file():
                raise SystemExit(f"Missing test annotation (empty file required for negatives): {label_path}")
            image_id = str(path.resolve())
            expected_by_image[image_id] = []
            predicted_by_image[image_id] = []
            for item in read_txt(label_path, w, h):
                if not 0 <= item["class"] < len(classes):
                    raise SystemExit(f"Invalid label class ID in {label_path}: {item['class']}")
                expected_by_image[image_id].append((classes[item["class"]], item["bbox"]))
            for det in detections:
                if det.weak or det.class_name not in args.classes:
                    continue
                predicted_by_image[image_id].append((
                    det.class_name,
                    [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2],
                    det.confidence,
                ))
        metrics = load_metrics()
        tp, fp, fn, _ = metrics.match_images(expected_by_image, predicted_by_image, 0.5)
        map50 = metrics.mean_ap_images(expected_by_image, predicted_by_image, classes, [0.5])
        map5095 = metrics.mean_ap_images(
            expected_by_image, predicted_by_image, classes,
            [round(0.5 + i * 0.05, 2) for i in range(10)],
        )
        matrix = metrics.confusion_matrix_images(expected_by_image, predicted_by_image, classes, 0.5)
        print(f"\n{name} ({weights}): {statistics.mean(lat):.0f} ms/frame")
        for cls in args.classes:
            precision, recall, f1 = metrics.precision_recall_f1(
                tp.get(cls, 0), fp.get(cls, 0), fn.get(cls, 0))
            print(f"  {cls}: P={precision:.2f} R={recall:.2f} F1={f1:.2f} "
                  f"AP50={map50[cls]:.2f} "
                  f"(tp={tp.get(cls, 0)} fp={fp.get(cls, 0)} fn={fn.get(cls, 0)})")
        print(f"  mAP50={sum(map50.values()) / len(map50):.3f} "
              f"mAP50-95={sum(map5095.values()) / len(map5095):.3f}")
        print("  confusion (gt->pred): " + ", ".join(
            f"{gt}->{pred}={n}" for (gt, pred), n in sorted(matrix.items())))
    print("\nAdopt the candidate only if it wins on test recall/false-positives, "
          "not on a single screenshot.")


if __name__ == "__main__":
    main()
