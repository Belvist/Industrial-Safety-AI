"""Threshold sweep for violation classes: v3 vs v3-clean on the fixed test.

Runs inference ONCE per model (all detections >= floor), then scores a grid
of (no_helmet, no_vest) thresholds against the same predictions — cheap and
exactly reproducible (no randomness; model SHAs + commit recorded in JSON).

Saves training/sweep_v3_v3clean/: sweep.csv (per combo per class P/R/AP/FP/FN),
sweep.json (inputs, best combos, verdict), sweep.png (recall/FP curves) and
review/ example crops (analysis ONLY — test images must never train V4).

Usage:
    .venv/bin/python training/threshold_sweep.py \
        --data datasets/ppe_v3_clean/data.yaml \
        --models models/ppe-v3.onnx models/ppe-v3-clean.onnx \
        --person-model models/person.onnx --out training/sweep_v3_v3clean
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'training'))

GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
FIXED_OTHER = {'person': 0.45, 'helmet': 0.50, 'vest': 0.50,
               'mask': 0.50, 'no_mask': 0.60}
FLOOR = 0.25
REVIEW_CAP = 12


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def main() -> None:
    import cv2
    import yaml

    import metrics
    from app.inference.local_onnx import LocalOnnxProvider

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--person-model', default=None)
    p.add_argument('--out', default='training/sweep_v3_v3clean')
    p.add_argument('--grid', nargs='+', type=float, default=GRID)
    args = p.parse_args()

    data_path = Path(args.data).resolve()
    dataset = yaml.safe_load(data_path.read_text())
    yaml_root = Path(dataset.get('path') or data_path.parent)
    if not yaml_root.is_absolute():
        yaml_root = (data_path.parent / yaml_root).resolve()
    images = []
    for entry in ([dataset['test']] if isinstance(dataset['test'], str) else dataset['test']):
        images += sorted(p for p in (yaml_root / entry).glob('*')
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    print(f'Test images: {len(images)}', flush=True)

    gt_by_image, sizes = {}, {}
    for img_path in images:
        lab = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
        items = []
        for line in lab.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f'Bad label line in {lab}')
            c, vals = int(parts[0]), [float(v) for v in parts[1:]]
            x, y, w, h = vals
            items.append((dataset['names'][c], [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
        gt_by_image[img_path.name] = items
        frame = cv2.imread(str(img_path))
        sizes[img_path.name] = (frame.shape[1], frame.shape[0])

    out = Path(args.out)
    (out / 'review' / 'fn_no_helmet').mkdir(parents=True, exist_ok=True)
    (out / 'review' / 'fp_no_helmet').mkdir(parents=True, exist_ok=True)
    (out / 'review' / 'fn_no_vest').mkdir(parents=True, exist_ok=True)
    (out / 'review' / 'fp_no_vest').mkdir(parents=True, exist_ok=True)
    (out / 'review' / 'README_DO_NOT_TRAIN.md').write_text(
        '# Analysis only\n\nThese crops come from the held-out TEST split. '
        'Never add them (or their source frames) to any training set — '
        'that would leak the test into training and void all metrics.\n')

    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, cwd=ROOT,
                                check=True).stdout.strip()
    except Exception:
        commit = 'unknown'

    rows = []
    raw_by_model = {}
    for weights in args.models:
        provider = LocalOnnxProvider(
            weights, 0.4, 2,
            person_model_path=args.person_model,
            thresholds={**FIXED_OTHER, 'no_helmet': 0.0, 'no_vest': 0.0},
            floor=FLOOR, roi_mode='off',
        )
        provider.load()
        # Raw detections (weak flags ignored here: floor keeps >= 0.25).
        raw: dict[str, list] = {}
        for img_path in images:
            frame = cv2.imread(str(img_path))
            raw[img_path.name] = [(d.class_name, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2], d.confidence)
                                  for d in provider.predict_sync(frame)]
        raw_by_model[Path(weights).stem] = raw
        tag = Path(weights).stem
        for t_h in args.grid:
            for t_v in args.grid:
                pred = {}
                for name, dets in raw.items():
                    kept = []
                    for cls, box, conf in dets:
                        limit = {'no_helmet': t_h, 'no_vest': t_v}.get(cls, FIXED_OTHER.get(cls, 0.4))
                        if conf >= limit:
                            kept.append((cls, box, conf))
                    pred[name] = kept
                ap = metrics.mean_ap_images(gt_by_image, pred,
                                            ['no_helmet', 'no_vest'], [0.5])
                cm = metrics.confusion_matrix_images(gt_by_image, pred,
                                                     ['no_helmet', 'no_vest'], 0.5)
                for cls in ('no_helmet', 'no_vest'):
                    tp = cm.get((cls, cls), 0)
                    fp = cm.get(('background', cls), 0)
                    fn = cm.get((cls, 'background'), 0)
                    prec = tp / (tp + fp) if tp + fp else 0.0
                    rec = tp / (tp + fn) if tp + fn else 0.0
                    rows.append({'model': tag, 't_no_helmet': t_h, 't_no_vest': t_v,
                                 'class': cls, 'P': round(prec, 4), 'R': round(rec, 4),
                                 'AP50': round(ap[cls], 4), 'TP': tp, 'FP': fp, 'FN': fn})
        print(f'{tag}: scored {len(args.grid) ** 2} combos', flush=True)

    with open(out / 'sweep.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def best(model, cls, metric='R', fp_cap=None):
        cands = [r for r in rows if r['model'] == model and r['class'] == cls]
        if fp_cap is not None:
            cands = [r for r in cands if r['FP'] <= fp_cap] or cands
        return max(cands, key=lambda r: (r[metric], r['AP50']))

    summary = {
        'inputs': {'models': {Path(w).stem: sha(Path(w)) for w in args.models},
                   'test_images': len(images), 'grid': args.grid,
                   'fixed_other_thresholds': FIXED_OTHER, 'floor': FLOOR,
                   'roi_mode': 'off', 'code_commit': commit},
        'at_production_060': {
            f'{m}/{c}': next(r for r in rows
                             if r['model'] == m and r['class'] == c
                             and r['t_no_helmet'] == 0.6 and r['t_no_vest'] == 0.6)
            for m in {Path(w).stem for w in args.models} for c in ('no_helmet', 'no_vest')},
        'best_recall': {
            f'{m}/{c}': best(Path(m).stem if False else m, c)
            for m in [Path(w).stem for w in args.models] for c in ('no_helmet', 'no_vest')},
        'verdict': ('see sweep.csv; recall at 0.60 vs FP cost per combo. '
                    'Event-level >=80% needs video measurement, not this table.'),
    }
    (out / 'sweep.json').write_text(json.dumps(summary, indent=2))
    _plot(rows, out)
    _review_images(images, sizes, out, gt_by_image, raw_by_model)
    print(f'sweep done: {out.resolve()}', flush=True)


def _plot(rows, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib missing: skip sweep.png')
        return
    models = sorted({r['model'] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, cls, metric in zip(axes.flat, ['no_helmet'] * 2 + ['no_vest'] * 2,
                               ['R', 'FP', 'R', 'FP']):
        for m in models:
            # Vary the class' own threshold, hold the other at 0.5.
            pts = sorted(
                (r[f"t_{cls}"], r[metric]) for r in rows
                if r['model'] == m and r['class'] == cls
                and r['t_no_helmet' if cls == 'no_vest' else 't_no_vest'] == 0.5)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=m)
        ax.set_title(f'{cls} {metric} vs own threshold (other=0.5)')
        ax.set_xlabel('threshold')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'sweep.png', dpi=110)


def _review_images(images, sizes, out: Path, gt_by_image, raw_by_model) -> None:
    """Annotated TEST frames for v3-clean @ production point (0.6/0.6).

    Green = matched GT, RED = missed GT (fn_*) or false alarm (fp_*).
    Analysis only — never train on these (see README_DO_NOT_TRAIN.md).
    """
    import cv2

    import metrics

    tag = 'ppe-v3-clean' if 'ppe-v3-clean' in raw_by_model else sorted(raw_by_model)[-1]
    raw = raw_by_model[tag]
    limits = {**FIXED_OTHER, 'no_helmet': 0.60, 'no_vest': 0.60}
    saved = {}
    by_name = {p.name: p for p in images}
    for name in sorted(raw):
        frame = cv2.imread(str(by_name[name]))
        if frame is None:
            continue
        w, h = sizes[name]
        preds = [(c, b, s) for c, b, s in raw[name]
                 if s >= limits.get(c, 0.4)]
        for cls, folder in (('no_helmet', 'fn_no_helmet'), ('no_vest', 'fn_no_vest')):
            for _, gt_box in [g for g in gt_by_image[name] if g[0] == cls]:
                hit = any(p[0] == cls and metrics.iou(p[1], gt_box) >= 0.5
                          for p in sorted(preds, key=lambda p: -p[2]))
                if not hit and saved.get(folder, 0) < REVIEW_CAP:
                    vis = frame.copy()
                    x1, y1, x2, y2 = (int(gt_box[0] * w), int(gt_box[1] * h),
                                      int(gt_box[2] * w), int(gt_box[3] * h))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 240), 2)
                    cv2.putText(vis, f'MISSED {cls}', (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 240), 2)
                    cv2.imwrite(str(out / 'review' / folder / f'{tag}__{name}'), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                    saved[folder] = saved.get(folder, 0) + 1
        taken: set[int] = set()
        for cls, folder in (('no_helmet', 'fp_no_helmet'), ('no_vest', 'fp_no_vest')):
            for pi, (pc, pb, ps) in enumerate(sorted(
                    [p for p in preds if p[0] == cls], key=lambda p: -p[2])):
                if pi in taken:
                    continue
                hit = any(g[0] == cls and metrics.iou(pb, g[1]) >= 0.5
                          for g in gt_by_image[name])
                if not hit and saved.get(folder, 0) < REVIEW_CAP:
                    vis = frame.copy()
                    x1, y1, x2, y2 = (int(pb[0] * w), int(pb[1] * h),
                                      int(pb[2] * w), int(pb[3] * h))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 60, 240), 2)
                    cv2.putText(vis, f'FALSE {cls} {ps:.2f}', (x1, max(16, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 240), 2)
                    cv2.imwrite(str(out / 'review' / folder / f'{tag}__{name}'), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
                    saved[folder] = saved.get(folder, 0) + 1
                    taken.add(pi)
    print('review examples:', saved, flush=True)


if __name__ == '__main__':
    main()
