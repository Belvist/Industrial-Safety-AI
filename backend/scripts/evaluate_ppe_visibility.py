"""Diagnostic real-model responses against provisional visual labels. Not accuracy acceptance.

Ground-truth rectangles are only used AFTER inference to match/evaluate results;
ROI inputs come exclusively from the actual person detector. No Runtime, no
injected detections/incidents and no synthetic image resizing are used.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--video-dir", type=Path, required=True)
p.add_argument("--labels", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
sys.path.insert(0, str(args.repo / "backend"))
from app.core.config import Settings  # noqa: E402
from app.domain.models import BoundingBox  # noqa: E402
from app.inference.local_onnx import LocalOnnxProvider  # noqa: E402
from app.services.engine import (  # noqa: E402
    SimpleIoUTracker,
    assign_detections,
    evaluate_assigned,
    iou,
)
from app.services.independent_ppe import observation_quality  # noqa: E402
from app.services.schemas import CameraConfig  # noqa: E402

labels = json.loads(args.labels.read_text())
args.output.mkdir(parents=True, exist_ok=False)
cfg = Settings(_env_file=None)
for name, src in labels["sources"].items():
    assert hashlib.sha256((args.video_dir / name).read_bytes()).hexdigest() == src["sha256"]
common = dict(
    model_path=str(args.repo / "models/ppe.onnx"),
    person_model_path=str(args.repo / "models/person.onnx"),
    thresholds=cfg.class_thresholds,
    floor=cfg.display_floor,
    confidence=cfg.confidence_threshold,
    threads=cfg.inference_threads,
)
provider = LocalOnnxProvider(**common, roi_mode="auto")
provider.load()
config = CameraConfig(helmet=True, vest=True)
rows = []
outputs = {}
for video, t in sorted({(r["video"], r["seconds"]) for r in labels["samples"]}):
    c = cv2.VideoCapture(str(args.video_dir / video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    fps = c.get(cv2.CAP_PROP_FPS)
    c.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = c.read()
    assert ok
    frame_index = int(c.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    (h, w) = frame.shape[:2]
    c.release()
    # Only the real person detector supplies model ROIs. Annotation data is not consulted here.
    people = provider.predict_people(frame)
    actual = {}
    for mode in ["full_frame", "roi_auto"]:
        det = (
            provider._predict_single(frame)
            if mode == "full_frame"
            else provider.predict_ppe(frame, people)
        )
        tracks = SimpleIoUTracker().update(people, t)
        assigned, _ = assign_detections(tracks, det, config)
        evaluate_assigned(tracks, assigned, config)
        quality = observation_quality(frame, tracks)
        actual[mode] = {"tracks": tracks, "quality": quality, "detections": det}
    outputs[f"{video}:{t}"] = {
        m: {
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": d.confidence,
                    "weak": d.weak,
                    "box": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                }
                for d in data["detections"]
            ],
            "people": [
                {"id": x.id, "box": [x.bbox.x1, x.bbox.y1, x.bbox.x2, x.bbox.y2], "ppe": x.ppe}
                for x in data["tracks"]
            ],
        }
        for m, data in actual.items()
    }
    # Manual labels are consulted only after both actual inference modes completed.
    selected = [r for r in labels["samples"] if r["video"] == video and r["seconds"] == t]
    overlay = frame.copy()
    for sample in selected:
        gt = BoundingBox(
            *(v / (640 if i % 2 == 0 else 360) for i, v in enumerate(sample["person"]))
        )

        def pixel(box, width=w, height=h):
            return [
                round(box[0] * width / 640, 1),
                round(box[1] * height / 360, 1),
                round(box[2] * width / 640, 1),
                round(box[3] * height / 360, 1),
            ]

        pb, hb, tb = map(pixel, [sample["person"], sample["head"], sample["torso"]])
        scale = min(640 / w, 640 / h)
        for mode, data in actual.items():
            matches = sorted(
                [(iou(gt, tr.bbox), tr) for tr in data["tracks"]], key=lambda x: x[0], reverse=True
            )
            match = matches[0] if matches and matches[0][0] >= 0.4 else None
            track = match[1] if match else None
            q = data["quality"]
            flags = q["tracks"].get(track.id, {}) if track else {}
            row = {
                "sample": sample["id"],
                "video": video,
                "source_seconds": frame_index / fps,
                "requested_seconds": t,
                "frame_index": frame_index,
                "mode": mode,
                "width": w,
                "height": h,
                "person_height_source_pixels": pb[3] - pb[1],
                "head_source_pixels": [hb[2] - hb[0], hb[3] - hb[1]],
                "torso_source_pixels": [tb[2] - tb[0], tb[3] - tb[1]],
                "gt_person_model_pixels": round((pb[3] - pb[1]) * scale, 1),
                "detected_person_model_pixels": flags.get("person_model_pixels"),
                "person_matched": bool(match),
                "person_iou": match[0] if match else None,
                "quality": {k: q[k] for k in ["sharpness", "luminance"]},
                "heuristic_hold": not flags.get("ppe_observable", False) if track else None,
                "heuristic_reasons": flags.get("reasons", []),
                "helmet_gt": sample["helmet"],
                "vest_gt": sample["vest"],
                "helmet_pred": track.ppe.get("helmet", "unknown") if track else "unknown",
                "vest_pred": track.ppe.get("vest", "unknown") if track else "unknown",
                "failure_reason": "UNKNOWN: no causal attribution from a model miss",
                "annotation_note": sample["note"],
            }
            for key in ["helmet", "vest"]:
                truth = row[key + "_gt"]
                pred = row[key + "_pred"]
                row[key + "_FP_violation"] = int(truth == "present" and pred == "missing")
                row[key + "_FN_violation"] = int(truth == "missing" and pred != "missing")
                row[key + "_TP_violation"] = int(truth == "missing" and pred == "missing")
                row[key + "_presence_miss"] = int(truth == "present" and pred != "present")
            row["failure_reason"] = (
                "UNKNOWN: model/assignment miss; no proven physical cause"
                if any(
                    row[k + "_gt"] != "unknown" and row[k + "_pred"] != row[k + "_gt"]
                    for k in ["helmet", "vest"]
                )
                else None
            )
            rows.append(row)
        for box, color in [(pb, (255, 160, 0)), (hb, (255, 0, 255)), (tb, (255, 0, 255))]:
            x1, y1, x2, y2 = map(round, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        overlay,
        "MANUAL EVALUATION REGIONS / NOT DETECTOR OUTPUT",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.imwrite(str(args.output / f"{Path(video).stem}-{t}-manual.jpg"), overlay)
summary = {
    mode: {
        "subjects": len(rr),
        "helmet_known_gt": sum(r["helmet_gt"] != "unknown" for r in rr),
        "helmet_UNKNOWN_results": sum(r["helmet_pred"] == "unknown" for r in rr),
        "vest_UNKNOWN_results": sum(r["vest_pred"] == "unknown" for r in rr),
        "person_misses": sum(not r["person_matched"] for r in rr),
        "heuristic_holds": sum(r["heuristic_hold"] is True for r in rr),
        "helmet_FP_violation": sum(r["helmet_FP_violation"] for r in rr),
        "helmet_FN_violation": sum(r["helmet_FN_violation"] for r in rr),
        "vest_FP_violation": sum(r["vest_FP_violation"] for r in rr),
        "vest_FN_violation": sum(r["vest_FN_violation"] for r in rr),
        "correct_status_despite_heuristic_hold": sum(
            r["heuristic_hold"]
            and any(
                r[k + "_gt"] != "unknown" and r[k + "_gt"] == r[k + "_pred"]
                for k in ["helmet", "vest"]
            )
            for r in rr
        ),
    }
    for mode in ["full_frame", "roi_auto"]
    for rr in [[r for r in rows if r["mode"] == mode]]
}
(args.output / "report.json").write_text(
    json.dumps(
        {
            "scope": labels["annotation_method"],
            "missing_class_note": "FP/FN refer to visual gear absence, not an actual offense in office/street footage. No incident was created.",
            "summary": summary,
            "rows": rows,
        },
        indent=2,
        ensure_ascii=False,
    )
)
(args.output / "actual-detections.json").write_text(json.dumps(outputs, indent=2))
print(json.dumps(summary, indent=2))
