"""Fine-tune a PPE detector. Requires data first — see datasets/README.md.

Usage:
    pip install ultralytics  # AGPL-3.0, see THIRD_PARTY.md before commercial use
    python training/train.py --data training/ppe-custom.yaml --model yolo11m.pt --epochs 100

Do NOT start long training without a labeled baseline and datasets/README.md
coverage. Compare against current weights with training/evaluate.py afterwards.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset.yaml from prepare_dataset.py")
    parser.add_argument("--model", default="yolo11m.pt", help="Base checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stopping patience (V4 spec: 15).")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="Ultralytics device (e.g. mps, cpu). Default: auto.")
    parser.add_argument("--project", default="training/runs")
    parser.add_argument("--name", default="ppe-custom")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics  # AGPL-3.0, check THIRD_PARTY.md first")

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, seed=args.seed, workers=args.workers,
        device=args.device,
        project=args.project, name=args.name, verbose=True,
    )
    print("Best weights: training/runs/ppe-custom/weights/best.pt")
    print("Next: training/export_onnx.py, then training/evaluate.py vs current weights.")


if __name__ == "__main__":
    main()
