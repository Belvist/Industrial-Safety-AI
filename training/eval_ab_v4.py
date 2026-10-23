"""Reproducible V3-clean vs V4 A/B on one fixed test split.

Every number in the report traces to: weight SHAs, git commit SHA, exact
per-class thresholds, dual-pipeline providers. No hand-waving.

Compares --base vs --candidate on --data test images (5 shared classes;
mask/no_mask GT excluded from scoring, never remapped). Reports per-class
TP/FP/FN + P/R/F1 @ IoU 0.5, latency, and renders EVERY error frame for the
violation classes (missed GT = FN sheets, false alarms = FP sheets, GT green
vs prediction red). Also embeds the independent-video incident summary when
--video-report points at one.

Usage (repo root, .venv):
    .venv/bin/python training/eval_ab_v4.py \\
        --data datasets/ppe_v3_clean/data.yaml \\
        --base models/ppe-v3-clean.onnx --candidate models/ppe-v4.onnx \\
        --person-model models/person.onnx --out training/eval_ab_v4_final

Output (all pushed to GitHub, no heavy binaries):
    REPORT.md, results.json, results.csv, fn_*.jpg / fp_*.jpg sheets,
    error_frames.json (every miss/alarm by filename).
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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

NAMES7 = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest', 'mask', 'no_mask']
KEEP = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest']
VIOL = ['no_helmet', 'no_vest']
IOU = 0.5


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
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, cwd=ROOT).stdout.strip()
    except Exception:
        return 'unknown'


def main() -> None:
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--person-model', default='models/person.onnx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--video-report', default=None,
                    help='Optional incidents.json from video scoring to embed.')
    args = ap.parse_args()
    metrics = load_metrics()

    import yaml
    data = yaml.safe_load(Path(args.data).read_text())
    names = list(data['names'])
    assert len(names) == 7, names
    root = Path(data.get('path') or Path(args.data).parent)
    if not root.is_absolute():
        root = (Path(args.data).parent / root).resolve()
    entries = data['test'] if isinstance(data['test'], list) else [data['test']]
    images: list[Path] = []
    for e in entries:
        images += sorted(p for p in (root / e).resolve().glob('*')
                         if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    assert images, 'no test images'

    thr = dict(settings.class_thresholds)
    fixed_thr = {k: thr[k] for k in KEEP}
    print('thresholds:', fixed_thr)

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
    gt_counts = {c: sum(1 for v in expected.values() for n, _ in v if n == c) for c in KEEP}
    print('GT:', gt_counts, f'({len(images)} images)')

    out = Path(args.out)
    (out / 'sheets').mkdir(parents=True, exist_ok=True)
    rows, error_index = [], {}
    for tag, wpath in (('base', args.base), ('candidate', args.candidate)):
        prov = LocalOnnxProvider(wpath, 0.4, 2, person_model_path=args.person_model,
                                 thresholds=thr, floor=0.25, roi_mode='off')
        prov.load()
        pred, ms = {}, []
        for p in images:
            fr = cv2.imread(str(p))
            t0 = time.perf_counter()
            dets = prov.predict_sync(fr)
            ms.append((time.perf_counter() - t0) * 1000)
            pred[p.name] = [(d.class_name, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                             d.confidence) for d in dets
                            if not d.weak and d.class_name in KEEP]
        tp, fp, fn, _ = metrics.match_images(expected, pred, IOU)
        ms_sorted = sorted(ms)
        row = {'model': tag, 'weights': wpath,
               'weights_sha256': sha256(Path(wpath)),
               'ms_mean': round(sum(ms) / len(ms), 1),
               'ms_p95': round(ms_sorted[int(len(ms) * 0.95)], 1)}
        for c in KEEP:
            pr, rc, f1 = metrics.precision_recall_f1(tp.get(c, 0), fp.get(c, 0), fn.get(c, 0))
            row[c] = {'tp': tp.get(c, 0), 'fp': fp.get(c, 0), 'fn': fn.get(c, 0),
                      'p': round(pr, 3), 'r': round(rc, 3), 'f1': round(f1, 3)}
        rows.append(row)

        # every error frame for the violation classes
        err = {'fn_images': {c: [] for c in VIOL}, 'fp_images': {c: [] for c in VIOL}}
        thumbs = {'fn': [], 'fp': []}
        for p in images:
            _t1, f1, n1, _ = metrics.match_images(
                {p.name: expected[p.name]}, {p.name: pred[p.name]}, IOU)
            fr = None
            for c in VIOL:
                if n1.get(c, 0):
                    err['fn_images'][c].append(p.name)
                    if len(thumbs['fn']) < 32:
                        fr = cv2.imread(str(p)) if fr is None else fr
                        draw_all(fr, expected[p.name], pred[p.name], p.name)
                        thumbs['fn'].append(cv2.resize(fr, (320, 240)))
                if f1.get(c, 0):
                    err['fp_images'][c].append(p.name)
                    if len(thumbs['fp']) < 32:
                        fr = cv2.imread(str(p)) if fr is None else fr
                        draw_all(fr, expected[p.name], pred[p.name], p.name)
                        thumbs['fp'].append(cv2.resize(fr, (320, 240)))
        error_index[tag] = err
        for kind in ('fn', 'fp'):
            if thumbs[kind]:
                sheet(thumbs[kind], out / 'sheets' / f'{tag}_{kind}.jpg')
        print(f'{tag}: ' + ' | '.join(
            f"{c} tp={row[c]['tp']} fp={row[c]['fp']} fn={row[c]['fn']} "
            f"P={row[c]['p']} R={row[c]['r']}" for c in KEEP), flush=True)

    commit = git_sha()
    (out / 'results.json').write_text(json.dumps(
        {'commit': commit, 'thresholds': fixed_thr, 'iou': IOU,
         'gt': gt_counts, 'n_images': len(images), 'models': rows}, indent=1))
    with (out / 'results.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'weights_sha256', 'class', 'tp', 'fp', 'fn', 'p', 'r',
                    'f1', 'ms_mean', 'ms_p95'])
        for r in rows:
            for c in KEEP:
                w.writerow([r['model'], r['weights_sha256'], c, r[c]['tp'], r[c]['fp'],
                            r[c]['fn'], r[c]['p'], r[c]['r'], r[c]['f1'],
                            r['ms_mean'], r['ms_p95']])
    (out / 'error_frames.json').write_text(json.dumps(error_index, indent=1))

    video = ''
    if args.video_report:
        rep = json.loads(Path(args.video_report).read_text())
        lines = ['\n## Independent video (no GT violations found — FP/latency only)\n']
        for clip in sorted(rep):
            for tag in ('v3c', 'v4'):
                r = rep[clip].get(tag, {})
                inc = ', '.join(f"{i['key'][1]}@{i['t_first']}s" for i in r.get('incidents', []))
                lines.append(f'- {clip} / {tag}: {r.get("n", 0)} ACTIVE FP'
                             f' ({inc or "none"}), {r.get("ms_mean")} ms/frame')
        video = '\n'.join(lines) + '\n'

    b, c = rows
    (out / 'REPORT.md').write_text(
        '# Reproducible A/B: base vs candidate (same images, same thresholds)\n\n'
        f'- commit: `{commit}`\n'
        f'- base: `{b["weights"]}` sha `{b["weights_sha256"]}`\n'
        f'- candidate: `{c["weights"]}` sha `{c["weights_sha256"]}`\n'
        f'- thresholds: `{fixed_thr}`, IoU {IOU}, dual pipeline '
        f'(person model `{args.person_model}`), ROI off\n'
        f'- test: `{args.data}` — {len(images)} images, GT {gt_counts}\n'
        '- mask/no_mask GT excluded from scoring (never remapped)\n\n'
        '| class | base tp/fp/fn (P/R) | candidate tp/fp/fn (P/R) |\n'
        '|---|---|---|\n' +
        ''.join(f'| {k} | {b[k]["tp"]}/{b[k]["fp"]}/{b[k]["fn"]} '
                f'({b[k]["p"]}/{b[k]["r"]}) | {c[k]["tp"]}/{c[k]["fp"]}/{c[k]["fn"]} '
                f'({c[k]["p"]}/{c[k]["r"]}) |\n' for k in KEEP) +
        f'\nLatency: base {b["ms_mean"]} ms (p95 {b["ms_p95"]}), '
        f'candidate {c["ms_mean"]} ms (p95 {c["ms_p95"]}) — CPU ONNX, this Mac.\n'
        '\nError sheets: `sheets/<model>_fn.jpg` (every miss, GT green) and '
        '`<model>_fp.jpg` (every false alarm, pred red); full lists in '
        '`error_frames.json`.\n' + video +
        '\nScope honesty: same-distribution stills (train/test share sources) — '
        'this measures in-distribution fit, NOT CCTV generalization. '
        'Derived torso boxes in V4 training are disclosed in V4_TRAIN_QA.md; '
        'CI tests cover pipeline code, not vision accuracy.\n')
    print(f'wrote {out}/REPORT.md + results + sheets', flush=True)


def draw_all(fr, exp_list, pred_list, name):
    h, w = fr.shape[:2]
    for n, (x1, y1, x2, y2) in exp_list:
        if n in VIOL:
            cv2.rectangle(fr, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                          (80, 200, 80), 2)
    for n, (x1, y1, x2, y2), _ in pred_list:
        if n in VIOL:
            cv2.rectangle(fr, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                          (60, 60, 240), 1)
    cv2.putText(fr, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def sheet(thumbs, path):
    rows = [thumbs[i:i + 4] for i in range(0, len(thumbs), 4)]
    canvas = np.zeros((len(rows) * 240, 1280, 3), np.uint8)
    for i, row in enumerate(rows):
        for j, t in enumerate(row):
            canvas[i * 240:(i + 1) * 240, j * 320:(j + 1) * 320] = t
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()
