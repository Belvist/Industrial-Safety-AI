"""Intrinsic person quality: PPE models vs the dedicated person detector.

Answers whether the separate person.onnx is load-bearing: runs each PPE
model with person_model_path=None (its OWN person boxes) and compares
person P/R/F1 against the dual-pipeline baseline on the same images.

Usage (repo root, .venv):
    .venv/bin/python training/eval_person.py --data datasets/ppe_v3_clean/data.yaml \\
        --models v3-clean:models/ppe-v3-clean.onnx v4:models/ppe-v4.onnx \\
                 v4cloud:models/ppe-v4cloud.onnx \\
        --person-model models/person.onnx --out training/eval_person
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

IOU = 0.5


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


def main() -> None:
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--models', nargs='+', required=True)
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
    thr = dict(settings.class_thresholds)

    expected: dict[str, list] = {}
    for p in images:
        lab = p.parent.parent / 'labels' / (p.stem + '.txt')
        cur = []
        for line in lab.read_text().splitlines():
            q = line.split()
            if len(q) == 5 and q[0] == '0':
                _, x, y, w, h = q[0], *map(float, q[1:])
                cur.append(('person', [x - w / 2, y - h / 2, x + w / 2, y + h / 2]))
        expected[p.name] = cur

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag, wpath in (m.split(':', 1) for m in args.models):
        for mode, extra in (('intrinsic', {'person_model_path': None}),
                            ('dual', {'person_model_path': args.person_model})):
            if mode == 'dual' and tag != args.models[0].split(':', 1)[0]:
                continue  # dedicated detector measured once (shared across models)
            prov = LocalOnnxProvider(wpath, 0.4, 2, thresholds=thr, floor=0.25,
                                     roi_mode='off', providers=['CPUExecutionProvider'],
                                     **extra)
            prov.load()
            pred, ms = {}, []
            for p in images:
                fr = cv2.imread(str(p))
                t0 = time.perf_counter()
                dets = prov.predict_sync(fr)
                ms.append((time.perf_counter() - t0) * 1000)
                pred[p.name] = [(d.class_name,
                                 [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                                 d.confidence)
                                for d in dets
                                if not d.weak and d.class_name == 'person']
            tp, fp, fn, _ = metrics.match_images(expected, pred, IOU)
            pr, rc, f1 = metrics.precision_recall_f1(
                tp.get('person', 0), fp.get('person', 0), fn.get('person', 0))
            rows.append({'model': tag, 'person_source': mode,
                         'tp': tp.get('person', 0), 'fp': fp.get('person', 0),
                         'fn': fn.get('person', 0), 'p': round(pr, 3),
                         'r': round(rc, 3), 'f1': round(f1, 3),
                         'ms_mean': round(sum(ms) / len(ms), 1)})
            print(f"{tag:8s} {mode:9s} P={pr:.3f} R={rc:.3f} "
                  f"tp={tp.get('person',0)} fp={fp.get('person',0)} "
                  f"fn={fn.get('person',0)} {sum(ms)/len(ms):.0f} ms", flush=True)
    (out / 'person.json').write_text(json.dumps(
        {'commit': git_sha(), 'mode': 'intrinsic-vs-dual person only',
         'rows': rows}, indent=1))
    print(f'wrote {out}/person.json', flush=True)


if __name__ == '__main__':
    main()
