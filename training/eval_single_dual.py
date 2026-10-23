"""SINGLE vs DUAL A/B for one PPE model: same weights/frames/thresholds/ROI.

Compares intrinsic person boxes (person_model_path=None) against the
dedicated person detector on: person TP/FP/FN, PPE TP/FP/FN (must be
identical — same net, verified not assumed), worker-band assignability of
every GT box under both person sources, and latency.

Usage (repo root, .venv):
    .venv/bin/python training/eval_single_dual.py --data datasets/ppe_v3_clean/data.yaml \\
        --weights models/ppe-v4cloud.onnx --person-model models/person.onnx \\
        --out training/eval_single_dual
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

NAMES7 = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest', 'mask', 'no_mask']
KEEP = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest']
IOU = 0.5
HEAD_BAND = (-0.15, 0.45)
TORSO_BAND = (0.15, 0.85)


def load_metrics():
    path = ROOT / 'training' / 'metrics.py'
    spec = importlib.util.spec_from_file_location('ppe_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], check=False,
                              capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def assignable(gt_box, person_boxes, cls):
    cx, cy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
    lo, hi = HEAD_BAND if cls in ('helmet', 'no_helmet') else TORSO_BAND
    for px1, py1, px2, py2 in person_boxes:
        w, h = px2 - px1, py2 - py1
        if h <= 0 or w <= 0:
            continue
        if px1 - 0.1 * w <= cx <= px2 + 0.1 * w and lo <= (cy - py1) / h <= hi \
                and box_iou(gt_box, [px1, py1, px2, py2]) > 0:
            return True
    return False


def main() -> None:
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--person-model', default='models/person.onnx')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    metrics = load_metrics()

    import yaml
    data = yaml.safe_load(Path(args.data).read_text())
    root = Path(data.get('path') or Path(args.data).parent)
    if not root.is_absolute():
        root = (Path(args.data).parent / root).resolve()
    entries = data['test'] if isinstance(data['test'], list) else [data['test']]
    images: list[Path] = []
    for e in entries:
        images += sorted(p for p in (root / e).resolve().glob('*')
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    assert images, 'no test images'
    thr = {k: dict(settings.class_thresholds)[k] for k in KEEP}

    expected: dict[str, list] = {}
    for p in images:
        lab = p.parent.parent / 'labels' / (p.stem + '.txt')
        cur = []
        for line in lab.read_text().splitlines():
            q = line.split()
            if len(q) == 5 and NAMES7[int(q[0])] in KEEP:
                _, x, y, w, h = q[0], *map(float, q[1:])
                cur.append((NAMES7[int(q[0])], [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
        expected[p.name] = cur

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = {}
    for mode, extra in (('single', {'person_model_path': None}),
                        ('dual', {'person_model_path': args.person_model})):
        prov = LocalOnnxProvider(args.weights, 0.4, 2, thresholds=thr, floor=0.25,
                                 roi_mode='off', providers=['CPUExecutionProvider'],
                                 **extra)
        prov.load()
        pred, persons, ms = {}, {}, []
        for p in images:
            fr = cv2.imread(str(p))
            t0 = time.perf_counter()
            dets = prov.predict_sync(fr)
            ms.append((time.perf_counter() - t0) * 1000)
            pred[p.name] = [(d.class_name, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                             d.confidence)
                            for d in dets if not d.weak and d.class_name in KEEP]
            persons[p.name] = [[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
                               for d in dets
                               if d.class_name == 'person' and not d.weak]
        modes[mode] = {'pred': pred, 'persons': persons, 'ms': ms}

    report: dict = {'commit': git_sha(), 'thresholds': thr, 'modes': {}}
    for mode in ('single', 'dual'):
        tp, fp, fn, _ = metrics.match_images(
            expected, modes[mode]['pred'], IOU)
        ms_sorted = sorted(modes[mode]['ms'])
        row: dict = {'ms_p50': round(ms_sorted[len(ms_sorted) // 2], 1),
                     'ms_p95': round(ms_sorted[int(len(ms_sorted) * 0.95)], 1),
                     'classes': {}, 'assignability': {}}
        for c in KEEP:
            pr, rc, f1 = metrics.precision_recall_f1(
                tp.get(c, 0), fp.get(c, 0), fn.get(c, 0))
            row['classes'][c] = {'tp': tp.get(c, 0), 'fp': fp.get(c, 0),
                                 'fn': fn.get(c, 0), 'p': round(pr, 3),
                                 'r': round(rc, 3), 'f1': round(f1, 3)}
        for c in ('helmet', 'no_helmet', 'vest', 'no_vest'):
            tot, ok = 0, 0
            for name, gt in expected.items():
                for gcls, gbox in gt:
                    if gcls != c:
                        continue
                    tot += 1
                    if assignable(gbox, modes[mode]['persons'][name], c):
                        ok += 1
            row['assignability'][c] = {'assignable': ok, 'total': tot,
                                       'rate': round(ok / tot, 3) if tot else 1.0}
        report['modes'][mode] = row
        print(f"{mode}: person tp={row['classes']['person']['tp']} "
              f"fp={row['classes']['person']['fp']} "
              f"fn={row['classes']['person']['fn']} | "
              f"ms p50={row['ms_p50']} p95={row['ms_p95']}", flush=True)
    for c in KEEP:
        s, d = report['modes']['single']['classes'][c], report['modes']['dual']['classes'][c]
        assert (s['tp'], s['fp'], s['fn']) == (d['tp'], d['fp'], d['fn']) or c == 'person', \
            f'PPE outputs diverged single vs dual for {c}: unsupported pipeline difference'
    print('PPE outputs identical single vs dual (person differs by design)', flush=True)
    (out / 'single_dual.json').write_text(json.dumps(report, indent=1))
    print(f'wrote {out}/single_dual.json', flush=True)


if __name__ == '__main__':
    main()
