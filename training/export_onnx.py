"""Export a trained checkpoint to the app's ONNX contract and verify it.

Checks: input 1x3x640x640, output YOLOv8-style (no NMS), class names mapping,
torch-vs-ONNX max abs diff on a sample image, SHA-256 sidecar for rollback.

Usage:
    python training/export_onnx.py training/runs/ppe-custom/weights/best.pt \\
        --out models/ppe-custom.onnx --sample data/samples/f0001.jpg
"""

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", default=None, help="Image for torch-vs-ONNX check")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics  # AGPL-3.0, check THIRD_PARTY.md first")

    out = Path(args.out)
    model = YOLO(args.checkpoint)
    exported = model.export(format="onnx", opset=args.opset, dynamic=False)
    Path(exported).replace(out)

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    assert list(shape) == [1, 3, 640, 640], f"Unexpected input {shape}"
    names = session.get_modelmeta().custom_metadata_map.get("names", "")
    print(f"Input OK {shape}; names metadata: {names[:120]}")
    (out.parent / (out.name + ".sha256")).write_text(f"{sha256(out)}  {out.name}\n")

    if args.sample:
        import ast

        import cv2
        from app.inference.local_onnx import LocalOnnxProvider

        frame = cv2.imread(args.sample)
        h, w = frame.shape[:2]
        # Identical preprocessing to production (centered letterbox, BGR->RGB).
        scale = min(640 / w, 640 / h)
        nw, nh = round(w * scale), round(h * scale)
        left, top = (640 - nw) // 2, (640 - nh) // 2
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
        tensor = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
        torch_boxes = model.predict(tensor, verbose=False)[0].boxes
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        probe = LocalOnnxProvider(str(out), 0.25, 1)
        names = ast.literal_eval(
            session.get_modelmeta().custom_metadata_map.get("names", "{}"))
        if isinstance(names, list):
            names = dict(enumerate(names))
        probe.classes = {int(k): str(v).lower() for k, v in names.items()}
        onnx_dets = probe.decode(raw, w, h, scale, left, top)
        tb = torch_boxes.data.cpu().numpy()
        matched, diffs = set(), []
        for det in [d for d in onnx_dets if not d.weak]:
            box = [det.bbox.x1 * w, det.bbox.y1 * h, det.bbox.x2 * w, det.bbox.y2 * h]
            best, best_iou = None, 0.5
            for i, row in enumerate(tb):
                if i in matched:
                    continue
                tx1, ty1, tx2, ty2 = row[:4]
                ix = max(0, min(box[2], tx2) - max(box[0], tx1))
                iy = max(0, min(box[3], ty2) - max(box[1], ty1))
                inter = ix * iy
                union = (box[2] - box[0]) * (box[3] - box[1]) + (tx2 - tx1) * (ty2 - ty1) - inter
                value = inter / union if union else 0
                if value >= best_iou:
                    best, best_iou = i, value
            if best is not None:
                matched.add(best)
                diffs.append(float(np.abs(np.array(box) - tb[best][:4]).max()))
        print(f"torch boxes: {len(tb)}, onnx decoded: {len(onnx_dets)}, "
              f"matched: {len(matched)}, max coord diff: {max(diffs) if diffs else 'n/a'} px")
        if len(matched) != len(tb) or len(matched) != len([d for d in onnx_dets if not d.weak]):
            raise SystemExit("PyTorch/ONNX detection count or alignment mismatch — export rejected.")
        if (len(tb) == 0 and len(onnx_dets) == 0):
            raise SystemExit("Parity sample has no detections; use a known-positive sample.")
        if diffs and max(diffs) > 5.0:
            raise SystemExit("PyTorch/ONNX mismatch beyond 5 px — do not ship this export.")
    print(f"Wrote {out} + .sha256. Register in THIRD_PARTY.md; keep old weights for rollback.")


if __name__ == "__main__":
    main()
