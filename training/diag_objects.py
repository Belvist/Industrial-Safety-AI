"""Object-level diagnostics for V3-clean vs V4 + ROI/provider matrix.

Part 1 (matrix): same 124 images, both models x ROI{off,auto} x providers
{CPU,CoreML-if-available} at fixed production thresholds. Records latency
(p50/p95/FPS) and P/R/F1 per class. Never mixes configurations.

Part 2 (objects, reference config = ROI off + CPU, same as published A/B):
for EVERY GT box: best same-class candidate (conf/IoU/weak), best any-class
candidate, person coverage, worker-band assignability, and one cause for each
FN. For every FP: best GT overlap, class confusion, cause. Asserts that
object-level sums equal match_images() totals. Renders one crop per
violation-class error object (GT green, best prediction red).

FN cause priority: wrong_class > bad_iou > below_threshold > no_person >
no_assign > no_detection. FP causes: duplicate > wrong_class >
mislocalized > background. Full definitions in DIAGNOSTIC_REPORT.md.

Usage (repo root, .venv):
    .venv/bin/python training/diag_objects.py --data datasets/ppe_v3_clean/data.yaml \\
        --base models/ppe-v3-clean.onnx --candidate models/ppe-v4.onnx \\
        --person-model models/person.onnx --out training/diag_v4
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
HEAD_BAND = (-0.15, 0.45)
TORSO_BAND = (0.15, 0.85)


def load_metrics():
    path = ROOT / 'training' / 'metrics.py'
    spec = importlib.util.spec_from_file_location('ppe_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


METRICS = load_metrics()


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


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def band(cls):
    return HEAD_BAND if cls in ('helmet', 'no_helmet') else TORSO_BAND


def assignable(gt_box, person_boxes, cls):
    """Emulate assign_detections band rule with person *detections* as tracks.

    Still images have no tracker IDs; this answers 'would the band rule attach
    a detection here to a worker'. Documented approximation, not the tracker.
    """
    cx, cy = (gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2
    lo, hi = band(cls)
    for px1, py1, px2, py2 in person_boxes:
        w, h = px2 - px1, py2 - py1
        if h <= 0 or w <= 0:
            continue
        rel = (cy - py1) / h
        if px1 - 0.1 * w <= cx <= px2 + 0.1 * w and lo <= rel <= hi \
                and box_iou(gt_box, [px1, py1, px2, py2]) > 0:
            return True
    return False


def main() -> None:
    import onnxruntime as ort
    from app.core.config import settings
    from app.inference.local_onnx import LocalOnnxProvider

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--person-model', default='models/person.onnx')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

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
    avail = ort.get_available_providers()
    providers = {'cpu': ['CPUExecutionProvider']}
    if 'CoreMLExecutionProvider' in avail:
        providers['coreml'] = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    print('providers available:', avail, flush=True)
    out = Path(args.out)
    (out / 'crops').mkdir(parents=True, exist_ok=True)

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

    # ---- Part 1: matrix ----
    matrix = []
    for tag, wpath in (('base', args.base), ('candidate', args.candidate)):
        for roi in ('off', 'auto'):
            for pname, plist in providers.items():
                try:
                    prov = LocalOnnxProvider(
                        wpath, 0.4, 2, person_model_path=args.person_model,
                        thresholds=dict(settings.class_thresholds), floor=0.25,
                        roi_mode=roi, providers=plist)
                    prov.load()
                    active = list(prov.active_providers)
                except Exception as exc:  # e.g. CoreML rejects the graph
                    matrix.append({'model': tag, 'roi': roi, 'provider': pname,
                                   'error': f'{type(exc).__name__}: {exc}'})
                    print(f'{tag}/{roi}/{pname}: FAILED {exc}', flush=True)
                    continue
                pred, ms = {}, []
                for p in images:
                    fr = cv2.imread(str(p))
                    t0 = time.perf_counter()
                    dets = prov.predict_sync(fr)
                    ms.append((time.perf_counter() - t0) * 1000)
                    pred[p.name] = [(d.class_name,
                                     [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                                     d.confidence)
                                    for d in dets if not d.weak and d.class_name in KEEP]
                tp, fp, fn, _ = METRICS.match_images(expected, pred, IOU)
                ms_sorted = sorted(ms)
                row = {'model': tag, 'roi': roi, 'provider': pname,
                       'active_providers': active,
                       'ms_p50': round(ms_sorted[len(ms_sorted) // 2], 1),
                       'ms_p95': round(ms_sorted[int(len(ms_sorted) * 0.95)], 1),
                       'fps': round(1000 / (sum(ms) / len(ms)), 2)}
                for c in KEEP:
                    pr, rc, f1 = METRICS.precision_recall_f1(
                        tp.get(c, 0), fp.get(c, 0), fn.get(c, 0))
                    row[c] = {'tp': tp.get(c, 0), 'fp': fp.get(c, 0), 'fn': fn.get(c, 0),
                              'p': round(pr, 3), 'r': round(rc, 3), 'f1': round(f1, 3)}
                matrix.append(row)
                print(f'{tag}/{roi}/{pname}: {row["ms_p50"]}ms p50, '
                      f'fps {row["fps"]}, active={active}', flush=True)
    (out / 'matrix.json').write_text(json.dumps(matrix, indent=1))

    # ---- Part 2: object-level on reference config (off/CPU) ----
    prov_cache = {}
    for tag, wpath in (('base', args.base), ('candidate', args.candidate)):
        prov = LocalOnnxProvider(
            wpath, 0.4, 2, person_model_path=args.person_model,
            thresholds=dict(settings.class_thresholds), floor=0.25,
            roi_mode='off', providers=['CPUExecutionProvider'])
        prov.load()
        per_image = {}
        for p in images:
            fr = cv2.imread(str(p))
            dets = prov.predict_sync(fr)
            per_image[p.name] = [
                {'class': d.class_name,
                 'box': [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
                 'conf': round(d.confidence, 4), 'weak': bool(d.weak)}
                for d in dets if d.class_name in KEEP]
        prov_cache[tag] = per_image

    objects = {'fn': [], 'fp': []}
    for tag in ('base', 'candidate'):
        per_image = prov_cache[tag]
        totals = {'tp': {}, 'fp': {}, 'fn': {}}
        for name, gt in expected.items():
            strong = [(d['class'], d['box'], d['conf'])
                      for d in per_image[name] if not d['weak']]
            persons = [d['box'] for d in per_image[name]
                       if d['class'] == 'person' and not d['weak']]
            # replicate metrics.match greedy order for attribution
            taken, order = set(), sorted(range(len(strong)), key=lambda i: -strong[i][2])
            match_of_pred: dict[int, int | None] = {}
            for i in order:
                cls, box, _ = strong[i]
                hit, best = None, IOU
                for gi, (gcls, gbox) in enumerate(gt):
                    if gi in taken or gcls != cls:
                        continue
                    v = box_iou(box, gbox)
                    if v >= best:
                        best, hit = v, gi
                match_of_pred[i] = hit
                if hit is None:
                    totals['fp'][cls] = totals['fp'].get(cls, 0) + 1
                else:
                    taken.add(hit)
                    totals['tp'][cls] = totals['tp'].get(cls, 0) + 1
            gt_taken = set(taken)
            for gi, (gcls, gbox) in enumerate(gt):
                if gi in gt_taken:
                    continue
                totals['fn'][gcls] = totals['fn'].get(gcls, 0) + 1
                same = [(d['conf'], d['box'], d['weak'])
                        for d in per_image[name] if d['class'] == gcls]
                other = [(d['class'], d['conf'], d['box'])
                         for d in per_image[name]
                         if not d['weak'] and d['class'] != gcls]
                best_same = max(same, key=lambda t: box_iou(t[1], gbox),
                                default=(0.0, None, True))
                best_other, best_other_iou = None, 0.0
                for c, cf, b in other:
                    v = box_iou(b, gbox)
                    if v > best_other_iou:
                        best_other, best_other_iou = (c, cf, b), v
                sic, sib, swk = best_same if best_same[1] is not None else (0.0, None, True)
                siou = box_iou(sib, gbox) if sib else 0.0
                thr_c = thr[gcls]
                person_cover = any(px1 <= (gbox[0] + gbox[2]) / 2 <= px2 and
                                   py1 <= (gbox[1] + gbox[3]) / 2 <= py2
                                   for px1, py1, px2, py2 in persons)
                assign = assignable(gbox, persons, gcls)
                if best_other and best_other_iou >= IOU:
                    cause = 'wrong_class'
                elif siou >= IOU:
                    cause = 'taken_by_other_gt'
                elif sic >= thr_c and siou < IOU:
                    cause = 'bad_iou'
                elif sib is not None and sic < thr_c:
                    cause = 'below_threshold'
                elif not person_cover:
                    cause = 'no_person'
                elif not assign:
                    cause = 'no_assign'
                else:
                    cause = 'no_detection'
                rec = {'model': tag, 'image': name, 'class': gcls,
                       'gt_box': [round(v, 4) for v in gbox],
                       'best_same': {'conf': sic, 'iou': round(siou, 3),
                                     'weak': bool(swk)} if sib else None,
                       'best_other': {'class': best_other[0], 'conf': best_other[1],
                                      'iou': round(best_other_iou, 3)}
                       if best_other else None,
                       'thr': thr_c, 'person_cover': person_cover,
                       'assignable': assign, 'cause': cause}
                objects['fn'].append(rec)
                if gcls in VIOL:
                    crop_error(out, images, name, gbox,
                               best_same_box=sib if sic >= 0.05 else None,
                               tag=f'{tag}_fn_{name}_{gi}')
            for i in order:
                if match_of_pred[i] is not None:
                    continue
                cls, box, conf = strong[i]
                best_gt, best_iou = None, 0.0
                for gi, (gcls, gbox) in enumerate(gt):
                    v = box_iou(box, gbox)
                    if v > best_iou:
                        best_gt, best_iou = (gcls, gi), v
                if best_gt and best_gt[0] == cls and best_iou >= IOU:
                    cause = 'duplicate'
                elif best_gt and best_gt[0] != cls and best_iou >= IOU:
                    cause = 'wrong_class'
                elif best_gt and best_gt[0] == cls and best_iou >= 0.3:
                    cause = 'mislocalized'
                else:
                    cause = 'background'
                persons = [d['box'] for d in per_image[name]
                           if d['class'] == 'person' and not d['weak']]
                rec = {'model': tag, 'image': name, 'class': cls,
                       'conf': conf, 'box': [round(v, 4) for v in box],
                       'best_gt': {'class': best_gt[0], 'iou': round(best_iou, 3)}
                       if best_gt else None,
                       'person_cover': any(
                           px1 <= (box[0] + box[2]) / 2 <= px2 and
                           py1 <= (box[1] + box[3]) / 2 <= py2
                           for px1, py1, px2, py2 in persons),
                       'cause': cause}
                objects['fp'].append(rec)
                if cls in VIOL:
                    crop_error(out, images, name, box, tag=f'{tag}_fp_{name}_{i}',
                               pred_box=True)
        # verify sums against match_images totals
        check_pred = {n: [(d['class'], d['box'], d['conf']) for d in per_image[n]
                          if not d['weak']] for n in expected}
        t2, f2, n2, _ = METRICS.match_images(expected, check_pred, IOU)
        assert t2 == totals['tp'] and f2 == totals['fp'] and n2 == totals['fn'], \
            f'{tag}: attribution sums diverge from match_images'
        print(f'{tag}: sums verified {sum(totals["fn"].values())} FN, '
              f'{sum(totals["fp"].values())} FP', flush=True)

    (out / 'objects_fn.json').write_text(json.dumps(objects['fn'], indent=1))
    (out / 'objects_fp.json').write_text(json.dumps(objects['fp'], indent=1))
    with (out / 'objects_fn.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'image', 'class', 'cause', 'best_same_conf',
                    'best_same_iou', 'best_same_weak', 'best_other', 'thr',
                    'person_cover', 'assignable'])
        for r in objects['fn']:
            b = r['best_same'] or {}
            o = r['best_other'] or {}
            w.writerow([r['model'], r['image'], r['class'], r['cause'],
                        b.get('conf'), b.get('iou'), b.get('weak'),
                        f"{o.get('class')}:{o.get('conf')}:{o.get('iou')}" if o else '',
                        r['thr'], r['person_cover'], r['assignable']])
    with (out / 'objects_fp.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'image', 'class', 'conf', 'cause', 'best_gt',
                    'person_cover'])
        for r in objects['fp']:
            g = r['best_gt'] or {}
            w.writerow([r['model'], r['image'], r['class'], r['conf'], r['cause'],
                        f"{g.get('class')}:{g.get('iou')}" if g else '',
                        r['person_cover']])

    from collections import Counter
    cause_fn = Counter((r['model'], r['class'], r['cause']) for r in objects['fn'])
    cause_fp = Counter((r['model'], r['class'], r['cause']) for r in objects['fp'])
    (out / 'causes.json').write_text(json.dumps(
        {'fn': {f'{m}/{c}/{k}': v for (m, c, k), v in sorted(cause_fn.items())},
         'fp': {f'{m}/{c}/{k}': v for (m, c, k), v in sorted(cause_fp.items())}},
        indent=1))
    print('causes FN:', dict(cause_fn), flush=True)
    print('causes FP:', dict(cause_fp), flush=True)
    (out / 'meta.json').write_text(json.dumps(
        {'commit': git_sha(), 'thresholds': thr, 'iou': IOU,
         'base_sha': sha256(Path(args.base)), 'candidate_sha': sha256(Path(args.candidate)),
         'config': 'roi off, CPU, dual pipeline, production thresholds'}, indent=1))
    print(f'done: {out}', flush=True)


def crop_error(out, images, name, box, best_same_box=None, tag='e', pred_box=False):
    src = next(p for p in images if p.name == name)
    fr = cv2.imread(str(src))
    h, w = fr.shape[:2]
    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    if pred_box:
        cv2.rectangle(fr, (x1, y1), (x2, y2), (60, 60, 240), 2)
    else:
        cv2.rectangle(fr, (x1, y1), (x2, y2), (80, 200, 80), 2)
        if best_same_box:
            b = best_same_box
            cv2.rectangle(fr, (int(b[0] * w), int(b[1] * h)),
                          (int(b[2] * w), int(b[3] * h)), (60, 60, 240), 1)
    pad = 60
    crop = fr[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
    cv2.putText(crop, tag, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(str(out / 'crops' / f'{tag}.jpg'), crop,
                [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == '__main__':
    main()
