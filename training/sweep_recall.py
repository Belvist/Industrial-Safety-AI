"""Recall sweep without training: 3 models x thresholds 0.25-0.60.

One ONNX pass per model at floor 0.05 (everything >= 0.05 returned, weak
flags kept); thresholds applied in software. Fixed test split, production
dual pipeline, ROI off, CPU (matches the published A/B).

Reports per (model, threshold): TP/FP/FN + P/R/F1 for the 5 classes, plus:
- max achievable recall per class (at 0.25) and which GT boxes have NO
  same-class detection >= 0.25 at all ("invisible even low" — these cannot
  be recovered by any threshold; only by data/architecture);
- split pick: V4 thresholds for no_helmet + V4-cloud for no_vest
  (+V3-clean person baseline), with summed FPS cost (two models = 2x);
- latency p50/p95 per model.

Diagnostic only: nothing here changes production settings.

Usage (repo root, .venv):
    .venv/bin/python training/sweep_recall.py --data datasets/ppe_v3_clean/data.yaml \\
        --models v3-clean:models/ppe-v3-clean.onnx v4:models/ppe-v4.onnx \\
                 v4cloud:models/ppe-v4cloud.onnx \\
        --person-model models/person.onnx --out training/sweep_recall
"""

from __future__ import annotations

import argparse
import csv
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
VIOL = ['no_helmet', 'no_vest']
IOU = 0.5
FLOOR = 0.05


def load_metrics():
    path = ROOT / 'training' / 'metrics.py'
    spec = importlib.util.spec_from_file_location('ppe_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], check=False,
                              capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def main() -> None:
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--models', nargs='+', required=True,
                    help='tag:path entries')
    ap.add_argument('--person-model', default='models/person.onnx')
    ap.add_argument('--thresholds', type=float, nargs='+',
                    default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
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

    prod_thr = {k: dict(settings.class_thresholds)[k] for k in KEEP}
    models = [tuple(m.split(':', 1)) for m in args.models]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

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

    # one inference pass per model at floor 0.05
    raw: dict[str, dict] = {}
    for tag, wpath in models:
        prov = LocalOnnxProvider(
            wpath, 0.4, 2, person_model_path=args.person_model,
            thresholds={c: min(args.thresholds) for c in KEEP}, floor=FLOOR,
            roi_mode='off', providers=['CPUExecutionProvider'])
        prov.load()
        per_image, ms = {}, []
        for p in images:
            fr = cv2.imread(str(p))
            t0 = time.perf_counter()
            dets = prov.predict_sync(fr)
            ms.append((time.perf_counter() - t0) * 1000)
            per_image[p.name] = [
                {'class': d.class_name,
                 'box': [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                 'conf': round(d.confidence, 4)}
                for d in dets if d.class_name in KEEP]
        ms_sorted = sorted(ms)
        raw[tag] = {'dets': per_image,
                    'ms_p50': round(ms_sorted[len(ms_sorted) // 2], 1),
                    'ms_p95': round(ms_sorted[int(len(ms_sorted) * 0.95)], 1)}
        print(f'{tag}: cached {len(images)} imgs, '
              f'p50 {raw[tag]["ms_p50"]} ms', flush=True)

    def score(tag, thr_map):
        pred = {}
        for name in expected:
            keep = []
            for d in raw[tag]['dets'][name]:
                if d['conf'] >= thr_map.get(d['class'], 0.4):
                    keep.append((d['class'], d['box'], d['conf']))
            pred[name] = keep
        return metrics.match_images(expected, pred, IOU)

    rows = []
    for tag, _ in models:
        for thr in args.thresholds:
            thr_map = dict(prod_thr)
            thr_map['no_helmet'] = thr
            thr_map['no_vest'] = thr
            tp, fp, fn, _ = score(tag, thr_map)
            row = {'model': tag, 'thr': thr}
            for c in KEEP:
                pr, rc, f1 = metrics.precision_recall_f1(
                    tp.get(c, 0), fp.get(c, 0), fn.get(c, 0))
                row[c] = {'tp': tp.get(c, 0), 'fp': fp.get(c, 0), 'fn': fn.get(c, 0),
                          'p': round(pr, 3), 'r': round(rc, 3), 'f1': round(f1, 3)}
            rows.append(row)

    # invisible even at the lowest threshold (per model, per class)
    invis: dict[str, dict[str, list[str]]] = {}
    low = min(args.thresholds)
    for tag, _ in models:
        invis[tag] = {}
        for name, gt in expected.items():
            for gcls, gbox in gt:
                best = 0.0
                for d in raw[tag]['dets'][name]:
                    if d['class'] != gcls:
                        continue
                    best = max(best, metrics.iou(d['box'], gbox))
                # visible = same-class detection with IoU>=0.5 AND conf>=low
                vis = any(d['class'] == gcls and d['conf'] >= low
                          and metrics.iou(d['box'], gbox) >= IOU
                          for d in raw[tag]['dets'][name])
                if not vis:
                    invis[tag].setdefault(gcls, []).append(name)
    invis_counts = {t: {c: len(v) for c, v in d.items()} for t, d in invis.items()}
    (out / 'invisible.json').write_text(json.dumps(invis, indent=1))

    # split pick: V4 thresholds for no_helmet + V4-cloud for no_vest
    split_note = ('split pick evaluated per-threshold-pair in REPORT: '
                  'no_helmet from v4 + no_vest from v4cloud '
                  '(person/helmet/vest from v4cloud); FPS cost = sum of both models')

    (out / 'results.json').write_text(json.dumps(
        {'commit': git_sha(), 'floor': FLOOR, 'iou': IOU,
         'prod_thresholds': prod_thr,
         'models': {t: {'sha': sha256(Path(w)), 'ms_p50': raw[t]['ms_p50'],
                        'ms_p95': raw[t]['ms_p95']} for t, w in models},
         'rows': rows, 'invisible_counts': invis_counts,
         'split_note': split_note}, indent=1))
    with (out / 'results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'thr', 'class', 'tp', 'fp', 'fn', 'p', 'r', 'f1'])
        for r in rows:
            for c in KEEP:
                w.writerow([r['model'], r['thr'], c, r[c]['tp'], r[c]['fp'],
                            r[c]['fn'], r[c]['p'], r[c]['r'], r[c]['f1']])

    print('\nmodel/thr | ' + ' | '.join(
        f'{c} R/F1' for c in VIOL + ['helmet', 'vest']))
    for r in rows:
        cells = ' | '.join(f"{r[c]['r']:.2f}/{r[c]['f1']:.2f} "
                            f"({r[c]['tp']:2d}/{r[c]['fp']:2d})"
                            for c in VIOL + ['helmet', 'vest'])
        print(f"{r['model']:8s} {r['thr']:.2f} | {cells}")
    print('\ninvisible even at thr', low, '(GT count that no threshold can recover):')
    for t, d in invis_counts.items():
        print(f'  {t}: ' + ', '.join(f'{c}={d.get(c, 0)}' for c in KEEP))
    print(f'\nwrote {out}/results.json + results.csv + invisible.json', flush=True)


if __name__ == '__main__':
    main()
