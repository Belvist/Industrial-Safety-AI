"""Part-based prototype: person -> head/torso crops -> PPE on crops + unknown.

No training, no prod changes, no new dependencies. For each strong person
detection, cut geometric head/torso crops and run the PPE model on the crop:

  head  = top 30% of person height, 15% width inset
  torso = 25-75% of person height, 15% width inset

Rules (documented v0 heuristics):
- head crop < 24 px                -> unknown (too small to judge)
- helmet conf >= thr               -> helmet (mapped to full-frame coords)
- elif no_helmet conf >= thr       -> no_helmet
- else                             -> unknown (SILENT, never a negative!)
- same for torso with vest/no_vest.

Why unknown matters: a hidden/tiny part must not become a false negative
vote. The baseline has no such state.

Compares against the box-level baseline (same model, same thresholds) with
identical metrics on the same images. Reports P/R/F1, unknown rate, and the
extra latency cost of crop inferences.

Usage (repo root, .venv):
    .venv/bin/python training/part_proto.py --data datasets/ppe_v3_clean/data.yaml \\
        --weights models/ppe-v4cloud.onnx --person-model models/person.onnx \\
        --out training/part_proto_out
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

NAMES7 = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest', 'mask', 'no_mask']
KEEP = ['person', 'helmet', 'no_helmet', 'vest', 'no_vest']
MIN_HEAD_PX = 24
MIN_TORSO_PX = 32


def load_metrics():
    path = ROOT / 'training' / 'metrics.py'
    spec = importlib.util.spec_from_file_location('ppe_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_person(box):
    """Geometric head/torso split of a normalized person box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ix1, ix2 = x1 + 0.15 * w, x2 - 0.15 * w
    head = [ix1, y1, ix2, y1 + 0.30 * h]
    torso = [ix1, y1 + 0.25 * h, ix2, y1 + 0.75 * h]
    return head, torso


def crop_frame(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(max(0, min(1, v)) * (w if i % 2 == 0 else h))
                      for i, v in enumerate(box))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def main() -> None:
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--person-model', default='models/person.onnx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--crop-thr', type=float, default=None,
                    help='Override thresholds on crops only (tests whether crop '
                         'confidences are merely depressed by domain shift).')
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
    crop_thr = dict(thr)
    if args.crop_thr is not None:
        for c in ('helmet', 'no_helmet', 'vest', 'no_vest'):
            crop_thr[c] = args.crop_thr
    prov = LocalOnnxProvider(
        args.weights, 0.4, 2, person_model_path=args.person_model,
        thresholds={c: 0.0 for c in
                    ['person', 'helmet', 'no_helmet', 'vest', 'no_vest',
                     'mask', 'no_mask']},
        floor=0.0, roi_mode='off',
        providers=['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    prov.load()
    print('active:', list(prov.active_providers), flush=True)

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

    base_pred, part_pred = {}, {}
    ms_base, ms_part, n_crops = [], [], 0
    ms_person_only, ms_ppe_full, ms_ppe_crop = [], [], []
    unknown = {'head': 0, 'head_small': 0, 'torso': 0, 'torso_small': 0, 'persons': 0}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for p in images:
        fr = cv2.imread(str(p))
        fh, fw = fr.shape[:2]
        t0 = time.perf_counter()
        dets = prov.predict_sync(fr)
        ms_base.append((time.perf_counter() - t0) * 1000)
        # split cost: person detector alone vs PPE full-frame alone
        # (PPE-only path; person result unused here — timing only)
        if prov.person_detector is not None:
            tp0 = time.perf_counter()
            prov.person_detector.predict_sync(fr)
            ms_person_only.append((time.perf_counter() - tp0) * 1000)
        te0 = time.perf_counter()
        prov._predict_single(fr)
        ms_ppe_full.append((time.perf_counter() - te0) * 1000)
        base_pred[p.name] = [
            (d.class_name, [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2], d.confidence)
            for d in dets
            if not d.weak and d.class_name in KEEP and d.confidence >= thr[d.class_name]]
        persons = [d for d in dets if d.class_name == 'person' and not d.weak
                   and d.confidence >= thr['person']]
        unknown['persons'] += len(persons)
        pp = []
        t1 = time.perf_counter()
        for d in persons:
            pb = [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
            pp.append(('person', pb, d.confidence))
            head, torso = split_person(pb)
            hw = (head[2] - head[0]) * fw
            tw = (torso[3] - torso[1]) * fh
            for box, classes, min_px, key in (
                    (head, ('helmet', 'no_helmet'), MIN_HEAD_PX, 'head'),
                    (torso, ('vest', 'no_vest'), MIN_TORSO_PX, 'torso')):
                px = hw if key == 'head' else tw
                if px < min_px:
                    unknown[key + '_small'] += 1
                    continue
                crop = crop_frame(fr, box)
                if crop is None:
                    unknown[key + '_small'] += 1
                    continue
                n_crops += 1
                # PPE-only path: person detector must NOT re-run per crop
                # (reviewer finding — predict_sync would redo it every time)
                tc0 = time.perf_counter()
                cdets = prov._predict_single(crop)
                ms_ppe_crop.append((time.perf_counter() - tc0) * 1000)
                # map crop detections back to full-frame coords
                mapped = []
                for cd in cdets:
                    if cd.weak or cd.class_name not in classes:
                        continue
                    fx1 = box[0] + cd.bbox.x1 * (box[2] - box[0])
                    fy1 = box[1] + cd.bbox.y1 * (box[3] - box[1])
                    fx2 = box[0] + cd.bbox.x2 * (box[2] - box[0])
                    fy2 = box[1] + cd.bbox.y2 * (box[3] - box[1])
                    mapped.append((cd.class_name, [fx1, fy1, fx2, fy2],
                                   round(cd.confidence, 4)))
                pos = [(c, b, cf) for c, b, cf in mapped
                       if c == classes[0] and cf >= crop_thr[c]]
                neg = [(c, b, cf) for c, b, cf in mapped
                       if c == classes[1] and cf >= crop_thr[c]]
                if pos and (not neg or max(x[2] for x in pos) >= max(x[2] for x in neg)):
                    pp.append(max(pos, key=lambda x: x[2]))
                elif neg:
                    pp.append(max(neg, key=lambda x: x[2]))
                else:
                    unknown[key] += 1
        ms_part.append((time.perf_counter() - t1) * 1000)
        part_pred[p.name] = pp

    def score(pred):
        tp, fp, fn, _ = metrics.match_images(expected, pred, 0.5)
        return tp, fp, fn

    btp, bfp, bfn = score(base_pred)
    ptp, pfp, pfn = score(part_pred)
    rep = {'thresholds': thr, 'crops': n_crops,
           'unknown': unknown, 'models': {}}
    print(f'\n{"class":9s} {"base P/R/F1 (tp fp fn)":32s} part P/R/F1 (tp fp fn)')
    for c in KEEP:
        bp, br, bf = metrics.precision_recall_f1(btp.get(c, 0), bfp.get(c, 0), bfn.get(c, 0))
        pp_, pr, pf = metrics.precision_recall_f1(ptp.get(c, 0), pfp.get(c, 0), pfn.get(c, 0))
        rep['models'][c] = {
            'base': {'tp': btp.get(c, 0), 'fp': bfp.get(c, 0), 'fn': bfn.get(c, 0),
                     'p': round(bp, 3), 'r': round(br, 3)},
            'part': {'tp': ptp.get(c, 0), 'fp': pfp.get(c, 0), 'fn': pfn.get(c, 0),
                     'p': round(pp_, 3), 'r': round(pr, 3)}}
        print(f'{c:9s} {bp:.3f}/{br:.3f}/{bf:.3f} '
              f'({btp.get(c,0):3d} {bfp.get(c,0):3d} {bfn.get(c,0):3d})   '
              f'{pp_:.3f}/{pr:.3f}/{pf:.3f} '
              f'({ptp.get(c,0):3d} {pfp.get(c,0):3d} {pfn.get(c,0):3d})')
    rep['latency'] = {
        'base_ms_mean': round(sum(ms_base) / len(ms_base), 1),
        'crop_stage_ms_mean': round(sum(ms_part) / len(ms_part), 1),
        'person_only_ms_mean': round(sum(ms_person_only) / len(ms_person_only), 1),
        'ppe_full_ms_mean': round(sum(ms_ppe_full) / len(ms_ppe_full), 1),
        'ppe_crop_ms_mean': round(sum(ms_ppe_crop) / len(ms_ppe_crop), 3),
        'crops_total': n_crops}
    print(f"\nlatency: base {rep['latency']['base_ms_mean']} ms/frame + "
          f"crop stage {rep['latency']['crop_stage_ms_mean']} ms/frame "
          f"({n_crops} crops)")
    print(f"split costs: person-only {rep['latency']['person_only_ms_mean']} ms, "
          f"PPE full-frame {rep['latency']['ppe_full_ms_mean']} ms, "
          f"PPE per-crop {rep['latency']['ppe_crop_ms_mean']} ms "
          f"(was: person re-ran on every crop)")
    print('unknown:', unknown)
    (out / 'part_report.json').write_text(json.dumps(rep, indent=1))
    print(f'wrote {out}/part_report.json', flush=True)


if __name__ == '__main__':
    main()
