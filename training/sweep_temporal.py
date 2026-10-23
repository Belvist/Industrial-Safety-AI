"""Temporal accumulation sweep over cached per-frame detections (no inference).

Reuses footage_v4/yt_eval/dets_<tag>_<clip>.json (all detections >= floor
0.25 with confidences). For each violation-threshold x temporal window:
re-runs Tracker + assign + TemporalEngine exactly like production and counts
ACTIVE incidents. All clips here are violation-free, so every ACTIVE is a
FALSE POSITIVE — this measures the FP/hour COST of low thresholds and wide
windows. The recall BENEFIT side comes from sweep_recall.py (box level);
event-level benefit on real violations needs labeled footage (none exists).

Grid: thr in {0.25, 0.35, 0.45, 0.60} applied to no_helmet/no_vest
(others fixed at production), windows (4,3) (6,4) (8,5), confirm 2.0s.

Usage: .venv/bin/python training/sweep_temporal.py --out training/sweep_temporal
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))

from app.domain.models import BoundingBox, Detection
from app.services.engine import (
    TemporalEngine,
    Tracker,
    assign_detections,
    cleared_assigned,
    evaluate_assigned,
)
from app.services.schemas import CameraConfig

DT = 0.2
CFG = CameraConfig(helmet=True, vest=True)


def run(dump_path: Path, thr: float, window: int, required: int):
    d = json.loads(dump_path.read_text())
    tracker, engine = Tracker(), TemporalEngine()
    n_active = 0
    first_t = None
    for step, (_name, dets) in enumerate(d['frames']):
        t = step * DT
        dd = []
        for cls, conf, box, _weak in dets:
            limit = thr if cls in ('no_helmet', 'no_vest') else {
                'person': 0.45, 'helmet': 0.50, 'vest': 0.50,
                'mask': 0.50, 'no_mask': 0.60, 'smoking': 0.55}.get(cls, 0.4)
            weak = conf < limit
            dd.append(Detection(cls, conf, BoundingBox(*box), weak=weak))
        tracks = tracker.update(dd, t)
        assigns, _ = assign_detections(tracks, dd, CFG)
        observed = evaluate_assigned(tracks, assigns, CFG)
        cleared = cleared_assigned(tracks, assigns, CFG)
        for action, _key, _st in engine.update(
                observed, t, 2.0, 2.0, window, required, cleared):
            if action == 'active':
                n_active += 1
                if first_t is None:
                    first_t = round(t, 1)
    return n_active, first_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clips-dir', default='footage_v4/yt_eval')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    minutes_total = 0.0
    rows = []
    for tag in ('v3c', 'v4', 'v4cloud'):
        dumps = sorted(Path(args.clips_dir).glob(f'dets_{tag}_*.json'))
        if not dumps:
            print(f'{tag}: no dumps (run /tmp/dump_dets.py first)', flush=True)
            continue
        for thr in (0.25, 0.35, 0.45, 0.60):
            for window, required in ((4, 3), (6, 4), (8, 5)):
                total, per_clip = 0, {}
                for dump in dumps:
                    n, first = run(dump, thr, window, required)
                    clip = dump.stem.replace(f'dets_{tag}_', '')
                    per_clip[clip] = {'n': n, 'first_t': first}
                    total += n
                minutes_total = sum(
                    len(json.loads(d.read_text())['frames']) * DT / 60 for d in dumps)
                rows.append({'model': tag, 'thr': thr, 'window': window,
                             'required': required, 'incidents': total,
                             'fp_per_hour': round(total / minutes_total * 60, 1)
                             if minutes_total else None,
                             'clips': per_clip})
                print(f"{tag} thr={thr} win={window}/{required}: "
                      f"{total} incidents (~{total / minutes_total * 60:.1f}/h)",
                      flush=True)
    (out / 'temporal.json').write_text(json.dumps(
        {'minutes_total': round(minutes_total, 1), 'rows': rows}, indent=1))
    with (out / 'temporal.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'thr', 'window', 'required', 'incidents', 'fp_per_hour'])
        for r in rows:
            w.writerow([r['model'], r['thr'], r['window'], r['required'],
                        r['incidents'], r['fp_per_hour']])
    print(f'wrote {out}/temporal.json ({minutes_total:.1f} min footage)', flush=True)


if __name__ == '__main__':
    main()
