"""Evaluate the detector over a folder of images.

Usage (repo root, .venv active):
    python backend/scripts/evaluate_folder.py data/samples [--labels labels.json] [--runs 1]

Without --labels reports honest detector statistics only (no ground truth
is invented): frames, latency/FPS, detections per class, mean/min confidence,
borderline ("weak") detections, unassigned PPE, worker count.

Labels JSON: {"images": {"name.jpg": [{"class": "person", "bbox": [x1,y1,x2,y2]}]}}
Labels CSV: header image,class,x1,y1,x2,y2. Coordinates normalized 0..1.
With labels additionally reports per-class precision/recall at IoU 0.5.

With --events EVENTS.json runs the full incident pipeline (tracker +
TemporalEngine consensus) over frames in filename order and compares
detected incidents against ground-truth events:
  {"events": [{"type": "no_helmet", "start_frame": 10, "end_frame": 50}]}
Reports matched/missed/false-positive incidents, detection-to-alert latency
in frames, and worker-ID stability proxy (distinct IDs per clip).

Worker-aware GT (recommended): each event names a stable worker id and
gives sparse worker boxes so an alarm on the WRONG person never counts:
  {"events": [{"type": "no_helmet", "worker": "w1",
               "start_frame": 10, "end_frame": 50,
               "boxes": {"10": [x1,y1,x2,y2], "30": [...]}}]}
Coordinates normalized 0..1. Events without boxes fall back to time-only
matching and are flagged worker_unverified in the report.
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0


def load_labels(path: Path) -> dict:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return data.get("images", data)
    rows: dict[str, list] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["image"], []).append(
                {
                    "class": row["class"],
                    "bbox": [float(row[k]) for k in ("x1", "y1", "x2", "y2")],
                }
            )
    return rows


def main():
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider
    from app.services.engine import Tracker, assign_detections
    from app.services.schemas import CameraConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--events", default=None,
                        help="GT events JSON for incident-level scoring")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frame timestamps step for the incident pipeline")
    parser.add_argument("--window", type=int, default=None,
                        help="Temporal consensus window (default: camera config)")
    parser.add_argument("--required", type=int, default=None,
                        help="Required positive votes in window (default: camera config)")
    parser.add_argument("--confirm", type=float, default=2.0,
                        help="Seconds of evidence before ACTIVE")
    parser.add_argument("--clear", type=float, default=2.0,
                        help="Seconds of counter-evidence before RESOLVED")
    args = parser.parse_args()

    images = sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"No images in {args.folder}")
    labels = load_labels(Path(args.labels)) if args.labels else {}

    import cv2

    provider = LocalOnnxProvider(
        str(settings.resolve(settings.model_path)),
        settings.confidence_threshold,
        settings.inference_threads,
        person_model_path=(
            str(settings.resolve(settings.person_model_path))
            if settings.use_separate_person_model
            else None
        ),
        providers=settings.onnx_provider_list,
        thresholds=settings.class_thresholds,
    )
    config = CameraConfig(helmet=True, vest=True, mask=True, smoking=True)
    tracker = Tracker()
    lat: list[float] = []
    per_class: Counter = Counter()
    conf_sum: dict[str, float] = {}
    weak = 0
    unassigned = 0
    workers: set[int] = set()
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for step, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"Skip undecodable: {path.name}")
            continue
        started = time.perf_counter()
        detections = provider.predict_sync(frame)
        lat.append((time.perf_counter() - started) * 1000)
        tracks = tracker.update(detections, step * 0.2)
        _, owners = assign_detections(tracks, detections, config)
        workers.update(t.id for t in tracks)
        for index, det in enumerate(detections):
            per_class[det.class_name] += 1
            conf_sum[det.class_name] = conf_sum.get(det.class_name, 0) + det.confidence
            limit = provider.threshold_for(det.class_name)
            if det.confidence < limit + 0.10:
                weak += 1
            if det.class_name != "person" and owners.get(index) is None:
                unassigned += 1
        expected = labels.get(path.name, [])
        matched = set()
        for det in detections:
            if det.weak:
                # Weak diagnostics never count as confirmed detections (T3).
                continue
            box = [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
            best, best_iou = None, 0.5
            for i, item in enumerate(expected):
                if i in matched or item["class"] != det.class_name:
                    continue
                value = iou(box, item["bbox"])
                if value >= best_iou:
                    best, best_iou = i, value
            if best is None:
                fp[det.class_name] += 1
            else:
                matched.add(best)
                tp[det.class_name] += 1
        for i, item in enumerate(expected):
            if i not in matched:
                fn[item["class"]] += 1

    print(f"Frames: {len(images)}, workers seen: {len(workers)}")
    print(f"Latency: mean {statistics.mean(lat):.0f} ms, "
          f"p95 {sorted(lat)[int(len(lat) * 0.95)]:.0f} ms, "
          f"{1000 / statistics.mean(lat):.1f} FPS")
    print("Detections/class:", dict(sorted(per_class.items())))
    print("Mean confidence: " + ", ".join(
        f"{k}={conf_sum[k] / per_class[k]:.2f}" for k in sorted(per_class)
    ))
    print(f"Weak (within 0.10 above threshold): {weak}, unassigned PPE: {unassigned}")
    if labels:
        print("Per-class P/R @ IoU 0.5 (labels provided):")
        for cls in sorted(set(tp) | set(fp) | set(fn)):
            precision = tp[cls] / max(1, tp[cls] + fp[cls])
            recall = tp[cls] / max(1, tp[cls] + fn[cls])
            print(f"  {cls}: P={precision:.2f} R={recall:.2f} "
                  f"(tp={tp[cls]} fp={fp[cls]} fn={fn[cls]})")
    if args.events:
        score_events(images, provider, args)


def _norm_boxes(boxes) -> dict[int, list[float]]:
    out = {}
    for k, v in (boxes or {}).items():
        try:
            out[int(k)] = [float(x) for x in v]
        except (ValueError, TypeError):
            continue
    return out


def _spatial_agreement(det_boxes: dict[int, list[float]],
                       gt_boxes: dict[int, list[float]],
                       start: int, end: int) -> tuple[bool, bool]:
    """Worker linkage between a detected incident and a GT event.

    Returns (agree, verifiable). Compares track boxes vs GT worker boxes on
    common annotated frames within the overlap: >=50% with IoU>=0.3 agrees.
    No common frames -> not verifiable (caller falls back to time-only and
    flags the match worker_unverified instead of inventing an identity).
    """
    frames = [f for f in gt_boxes if start <= f <= end and f in det_boxes]
    if not frames:
        return False, False
    hits = sum(1 for f in frames if iou(det_boxes[f], gt_boxes[f]) >= 0.3)
    return hits / len(frames) >= 0.5, True


def match_incidents(detected: list[dict], gt_events: list[dict]) -> dict:
    """Match detected incidents to GT events by type + time + worker.

    detected: [{type, start_frame, end_frame, worker, boxes:{frame: box}}]
    gt:       [{type, start_frame, end_frame, worker?, boxes?}]
    One-to-one greedy. A detection overlapping a GT event in type+time but
    failing the worker check is a false alarm (worker_mismatch), never a
    match — an alarm on the wrong person must not score.
    """
    matched, missed, false_alarms, latencies = [], [], [], []
    worker_mismatch, unverified = [], []
    taken = set()
    for di, det in enumerate(detected):
        hit, dt_unverified, dt_mismatch = None, False, False
        for gi, event in enumerate(gt_events):
            if gi in taken or event.get('type') != det.get('type'):
                continue
            if not (det['start_frame'] <= event['end_frame']
                    and event['start_frame'] <= det['end_frame']):
                continue
            start = max(det['start_frame'], event['start_frame'])
            end = min(det['end_frame'], event['end_frame'])
            gt_boxes = _norm_boxes(event.get('boxes'))
            if not gt_boxes:
                hit, dt_unverified = gi, True
                break
            agree, verifiable = _spatial_agreement(
                _norm_boxes(det.get('boxes')), gt_boxes, start, end)
            if agree:
                hit = gi
                break
            if not verifiable:
                hit, dt_unverified = gi, True
                break
            dt_mismatch = True
        if hit is None:
            false_alarms.append(di)
            if dt_mismatch:
                worker_mismatch.append(di)
        else:
            taken.add(hit)
            matched.append((di, hit))
            latencies.append(det['start_frame'] - gt_events[hit]['start_frame'])
            if dt_unverified:
                unverified.append(di)
    missed = [i for i in range(len(gt_events)) if i not in taken]
    return {'matched': matched, 'missed': missed, 'false_alarms': false_alarms,
            'latencies': latencies, 'worker_mismatch': worker_mismatch,
            'unverified': unverified}


def score_events(images, provider, args):
    """Incident-level scoring: GT events vs detected ACTIVE incidents."""
    import cv2

    from app.services.engine import (
        TemporalEngine,
        Tracker,
        assign_detections,
        cleared_assigned,
        evaluate_assigned,
    )
    from app.services.schemas import CameraConfig

    gt = json.loads(Path(args.events).read_text()).get("events", [])
    config = CameraConfig(helmet=True, vest=True, mask=True, smoking=True)
    window = args.window or config.temporal_window
    required = args.required or config.temporal_required
    tracker = Tracker()
    engine = TemporalEngine()
    dt = 1.0 / args.fps
    detected: list[dict] = []
    open_incidents: dict = {}
    id_frames: dict[int, int] = {}
    track_boxes: dict[int, dict[int, list[float]]] = {}
    for step, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        now = step * dt
        detections = provider.predict_sync(frame)
        tracks = tracker.update(detections, now)
        for track in tracks:
            id_frames[track.id] = id_frames.get(track.id, 0) + 1
            track_boxes.setdefault(track.id, {})[step] = [
                track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2]
        assignments, _ = assign_detections(tracks, detections, config)
        observed = evaluate_assigned(tracks, assignments, config)
        cleared = cleared_assigned(tracks, assignments, config)
        for action, key, _state in engine.update(
            observed, now, args.confirm, args.clear, window, required, cleared,
        ):
            if action == "active":
                open_incidents[key] = step
            elif action in {"resolved", "lost", "interrupted"} and key in open_incidents:
                start = open_incidents.pop(key)
                detected.append({
                    "type": key[1], "start_frame": start,
                    "end_frame": step, "outcome": action, "worker": key[0],
                    "boxes": {f: b for f, b in track_boxes.get(key[0], {}).items()
                              if start <= f <= step},
                })
    for key, start in open_incidents.items():
        detected.append({"type": key[1], "start_frame": start,
                         "end_frame": len(images) - 1, "outcome": "active",
                         "worker": key[0],
                         "boxes": {f: b for f, b in track_boxes.get(key[0], {}).items()
                                   if f >= start}})
    res = match_incidents(detected, gt)
    latencies = res["latencies"]
    unverified_det = set(res["unverified"])
    verified = [m for m in res["matched"] if m[0] not in unverified_det]
    print(f"\nIncident-level scoring ({len(gt)} GT events, {len(detected)} detected):")
    print(f"  params: window={window} required={required} "
          f"confirm={args.confirm}s clear={args.clear}s")
    print(f"  matched={len(res['matched'])} missed={len(res['missed'])} "
          f"false_alarms={len(res['false_alarms'])}")
    if res["worker_mismatch"]:
        print(f"  worker_mismatch (alarm on wrong person)={len(res['worker_mismatch'])}: "
              f"{res['worker_mismatch']}")
    if res["unverified"]:
        print(f"  worker_unverified (no GT boxes, time-only)={len(res['unverified'])}: "
              f"{res['unverified']}")
    print(f"  strict worker-verified matches={len(verified)} "
          f"(time-only matches excluded from the strict metric)")
    total_gt = len(gt) or 1
    print(f"  recall: worker-verified {len(verified)}/{len(gt)} = "
          f"{len(verified) / total_gt:.3f}; "
          f"time-only {len(res['matched'])}/{len(gt)} = "
          f"{len(res['matched']) / total_gt:.3f}")
    if latencies:
        print(f"  detection-to-alert: mean {statistics.mean(latencies):.1f} frames, "
              f"max {max(latencies)} frames")
    print(f"  worker-ID stability proxy: {len(id_frames)} distinct IDs, "
          f"frames/ID mean {statistics.mean(id_frames.values()):.1f}")


if __name__ == "__main__":
    main()
