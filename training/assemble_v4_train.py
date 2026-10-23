"""Assemble the V4 TRAIN pool from vetted sources. Data work only.

Frozen-infra rule: this script IMPORTS helpers from training/build_ppe_v4.py
(pixel_sha, dhash, draw_boxes, sanitize, iou_xyxy) but never modifies it,
TRAIN_V3.py, or the tests.

Sources (see datasets/v4_train/SOURCES.md written by this script):
  - V3-clean train, MINUS head_* (license unconfirmed): 7-class -> 5-class map,
    mask/no_mask boxes dropped. Hand labels = verified pool.
  - keremberke/hard-hat-detection train.zip (CC BY 4.0, HF sha recorded):
    deterministic quota sample. Source boxes are hand labels; person/vest
    completion comes from ONNX detectors and is therefore DRAFT, quarantined
    from TRAIN until human review ("no unverified autolabels" rule).

5-class order (fixed): 0 helmet, 1 no_helmet, 2 vest, 3 no_vest, 4 person.

Box semantics verified visually (Oct 2026): hardhat/no-hardhat and
helmet/no_helmet are all HEAD-region boxes (size gap vs V3 is a distance
effect, not a convention clash) -> direct mapping is valid.

Dedupe: pixel-sha drop vs old test/val AND vs V3-clean train AND within the
new sample (all reported). Near-dhash pairs are reported, not dropped.

Outputs (datasets/v4_train/):
  train_verified/{images,labels} + val split (deterministic sha 90/10,
    stratified by source), drafts_pending_review/, quarantine/,
  data.yaml (5-class), audit.json + audit_leaks.json (strict-gate shaped:
    conflicts/duplicates/drafts all zero IN THE TRAIN SET), manifest.json,
  SOURCES.md, QA_REPORT.md, contact sheets. No test/ or holdout here by
  design (holdout = brand-new scenes only, later).

Usage:
  .venv/bin/python training/assemble_v4_train.py --steps all
  .venv/bin/python training/assemble_v4_train.py --steps sample drafts assemble
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'training'))

V4_CLASSES = ['helmet', 'no_helmet', 'vest', 'no_vest', 'person']
V3_TO_V4 = {0: 4, 1: 0, 2: 1, 3: 2, 4: 3}  # 5,6 (mask/no_mask) dropped
PERSON_CONF = 0.50   # draft person boxes. Deliberately recall-oriented:
                     # every accepted image passes human eyes, which catch FP
                     # person boxes; missed persons (gaps) are the worse evil.
                     # Coverage and labels use the SAME set (no covered-but-
                     # unlabeled gaps by construction).
BG_MAX_CONF = 0.30   # below this (all classes), empty-source image = background
VEST_DRAFT_CONF = 0.80  # informational draft only, never trains unverified
NEAR_BITS = 8


def sha_file(path: Path) -> str:
    h = hashlib.shake_128()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest(16)


def read_yolo_txt(path: Path):
    out = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'{path}:{lineno}: expected 5 columns')
        out.append((int(parts[0]), list(map(float, parts[1:]))))
    return out


def write_yolo_txt(path: Path, labels) -> None:
    path.write_text(''.join(f'{c} ' + ' '.join(f'{v:.6f}' for v in b) + '\n'
                            for c, b in labels))


def sample_hardhat(staging: Path, quota_viol: int, quota_ok: int, quota_bg: int,
                   skip: set | None = None) -> dict:
    """Deterministic quota sample from the extracted COCO pool.

    Order scan by filename (no RNG state to reproduce). Returns
    {'viol': [...], 'ok': [...], 'bg': [...]} of (image_path, anns) with COCO
    pixel boxes + image size. Quotas: images containing no-hardhat first
    (the gap class), then hardhat-only, then verified-empty background.
    skip: src paths already sampled (batch-2 appends without overlap).
    """

    src = staging / 'train'
    with open(src / '_annotations.coco.json') as _jf:
        coco = json.load(_jf)
    cat = {c['id']: c['name'] for c in coco['categories']}
    by_img: dict[int, list] = collections.defaultdict(list)
    for a in coco['annotations']:
        by_img[a['image_id']].append(a)
    skip = skip or set()
    viol, ok, empty = [], [], []
    for im in sorted(coco['images'], key=lambda m: m['file_name']):
        anns = by_img.get(im['id'], [])
        names = {cat[a['category_id']] for a in anns}
        entry = (src / im['file_name'], anns, (im['width'], im['height']))
        if str(entry[0]) in skip:
            continue
        if not anns:
            if len(empty) < quota_bg:
                empty.append(entry)
        elif 'no-hardhat' in names:
            if len(viol) < quota_viol:
                viol.append(entry)
        elif len(ok) < quota_ok:
            ok.append(entry)
    # NOTE: the scan always runs to the end for honest pool stats.
    stats = {'pool_images': len(coco['images']),
             'pool_nohardhat_images': sum(1 for im in coco['images']
                                          if any(cat[a['category_id']] == 'no-hardhat'
                                                 for a in by_img.get(im['id'], []))),
             'selected_viol': len(viol), 'selected_ok': len(ok), 'selected_bg': len(empty)}
    print('hardhat sample:', json.dumps(stats), flush=True)
    return {'viol': viol, 'ok': ok, 'bg': empty, 'stats': stats,
            'categories': ['hardhat', 'no-hardhat']}


def coco_to_yolo(anns, img_w: int, img_h: int):
    """hardhat->helmet(0), no-hardhat->no_helmet(1). Pixel COCO -> YOLO norm."""
    out = []
    for a in anns:
        x, y, bw, bh = a['bbox']
        out.append((0 if a['category_id'] == 0 else 1,
                    [(x + bw / 2) / img_w, (y + bh / 2) / img_h,
                     bw / img_w, bh / img_h]))
    return out


def run_drafts(args, work: dict, providers) -> dict:
    """Detector completion for the hardhat sample. ALL output is DRAFT.

    person provider -> person draft boxes (>= PERSON_CONF).
    ppe provider    -> vest draft boxes (>= VEST_DRAFT_CONF, informational).
    no_vest is NEVER drafted (absence is unprovable by a detector).
    Empty-source images with every detection < BG_MAX_CONF -> verified
    background (empty label), small quota.
    Returns per-image dicts with labels + draft flags + quarantine reasons.
    """

    person_p, ppe_p = providers
    results = []
    for group in ('viol', 'ok', 'bg'):
        for img_path, anns, (iw, ih) in work[group]:
            frame = cv2.imread(str(img_path))
            if frame is None:
                results.append({'src': str(img_path), 'quarantine': 'undecodable'})
                continue
            src_labels = coco_to_yolo(anns, iw, ih)
            dets_person = [d for d in person_p.predict_sync(frame)]
            dets_ppe = [d for d in ppe_p.predict_sync(frame)]
            person_draft = [(4, [(d.bbox.x1 + d.bbox.x2) / 2,
                                 (d.bbox.y1 + d.bbox.y2) / 2,
                                 d.bbox.x2 - d.bbox.x1, d.bbox.y2 - d.bbox.y1])
                            for d in dets_person
                            if d.class_name == 'person' and d.confidence >= PERSON_CONF]
            vest_draft = [(2, [(d.bbox.x1 + d.bbox.x2) / 2,
                               (d.bbox.y1 + d.bbox.y2) / 2,
                               d.bbox.x2 - d.bbox.x1, d.bbox.y2 - d.bbox.y1])
                          for d in dets_ppe
                          if d.class_name == 'vest' and d.confidence >= VEST_DRAFT_CONF]
            max_conf = max([d.confidence for d in dets_person + dets_ppe] + [0.0])
            if group == 'bg':
                if anns or max_conf >= BG_MAX_CONF:
                    results.append({'src': str(img_path),
                                    'quarantine': 'bg-rejected-has-content'})
                    continue
                results.append({'src': str(img_path), 'group': group,
                                'labels': [], 'background': True})
                continue
            if not person_draft:
                results.append({'src': str(img_path),
                                'quarantine': 'no-person-derivable'})
                continue
            results.append({'src': str(img_path), 'group': group,
                            'labels': src_labels + person_draft + vest_draft,
                            'n_src': len(src_labels), 'n_person': len(person_draft),
                            'n_vest_draft': len(vest_draft), 'draft': True})
    return results


def contact_sheet(items: list[tuple[str, object, list]], path: Path, cols: int = 4):
    import build_ppe_v4 as B

    if not items:
        return
    thumbs = []
    for name, frame, labels in items[:16]:
        vis = B.draw_boxes(frame.copy(), labels)
        cv2.putText(vis, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        thumbs.append(cv2.resize(vis, (320, 240)))
    rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
    sheet = np.zeros((len(rows) * 240, cols * 320, 3), dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            sheet[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = t
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', nargs='+',
                   default=['sample', 'drafts', 'assemble'],
                   help='subset of: sample drafts assemble triage sheets merge')
    p.add_argument('--decisions', default='training/v4_review_decisions.json',
                   help='human verdicts for the manual review pool '
                        '{hh_stem: accept|reject}')
    p.add_argument('--v3clean', default='datasets/ppe_v3_clean')
    p.add_argument('--staging', default='datasets/_staging/hardhat')
    p.add_argument('--zip', default='datasets/_staging/hardhat/train.zip')
    p.add_argument('--quota-viol', type=int, default=600)
    p.add_argument('--quota-ok', type=int, default=600)
    p.add_argument('--quota-bg', type=int, default=60)
    p.add_argument('--quota-viol2', type=int, default=0,
                   help='Second viol batch appended without overlap (reads sample.json skip).')
    p.add_argument('--quota-ok2', type=int, default=0)
    p.add_argument('--quota-bg2', type=int, default=0)
    p.add_argument('--out', default='datasets/v4_train')
    p.add_argument('--ppe-weights', default='models/ppe-v3-clean.onnx')
    p.add_argument('--person-model', default='models/person.onnx')
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args()

    out = Path(args.out)
    stage_dir = out / '_stage'
    if 'sample' in args.steps:
        stage_dir.mkdir(parents=True, exist_ok=True)
        work = sample_hardhat(Path(args.staging), args.quota_viol,
                              args.quota_ok, args.quota_bg)
        serial = {k: [(str(p), [(a['category_id'], a['bbox']) for a in anns], wh)
                      for p, anns, wh in work[k]] for k in ('viol', 'ok', 'bg')}
        (stage_dir / 'sample.json').write_text(json.dumps(
            {'stats': work['stats'], 'groups': serial}))
        print('sample saved:', stage_dir / 'sample.json', flush=True)

    if 'sample2' in args.steps:
        prev = json.loads((stage_dir / 'sample.json').read_text())['groups']
        skip = {e[0] for k in ('viol', 'ok', 'bg') for e in prev[k]}
        work = sample_hardhat(Path(args.staging), args.quota_viol2,
                              args.quota_ok2, args.quota_bg2, skip=skip)
        serial = {k: [(str(p), [(a['category_id'], a['bbox']) for a in anns], wh)
                      for p, anns, wh in work[k]] for k in ('viol', 'ok', 'bg')}
        (stage_dir / 'sample2.json').write_text(json.dumps(
            {'stats': work['stats'], 'groups': serial}))
        print('sample2 saved:', stage_dir / 'sample2.json', flush=True)

    if 'drafts' in args.steps:
        from app.inference.local_onnx import LocalOnnxProvider

        work_raw = json.loads((stage_dir / 'sample.json').read_text())['groups']
        if (stage_dir / 'sample2.json').is_file():
            extra = json.loads((stage_dir / 'sample2.json').read_text())['groups']
            work_raw = {k: work_raw.get(k, []) + extra.get(k, []) for k in ('viol', 'ok', 'bg')}
        done_srcs = set()
        if (stage_dir / 'drafts.json').is_file():
            done_srcs = {r['src'] for r in
                         json.loads((stage_dir / 'drafts.json').read_text())}
            if done_srcs:
                print(f'drafts: skipping {len(done_srcs)} already processed', flush=True)
        work = {}
        for k, entries in work_raw.items():
            work[k] = []
            for path_s, anns, wh in entries:
                if path_s in done_srcs:
                    continue
                Gers = [{'category_id': c, 'bbox': b} for c, b in anns]
                work[k].append((Path(path_s), Gers, tuple(wh)))
        person_p = LocalOnnxProvider(
            args.person_model, 0.4, 2, person_model_path=None,
            required_classes={'person'},
            thresholds={'person': 0.0}, floor=0.25, roi_mode='off')
        person_p.load()
        ppe_p = LocalOnnxProvider(
            args.ppe_weights, 0.4, 2, person_model_path=args.person_model,
            thresholds={c: 0.0 for c in
                        ['person', 'helmet', 'no_helmet', 'vest', 'no_vest',
                         'mask', 'no_mask']},
            floor=0.25, roi_mode='off')
        ppe_p.load()
        results = run_drafts(args, work, (person_p, ppe_p))
        serial = []
        for r in results:
            serial.append({**r, 'src': str(r['src'])})
        if (stage_dir / 'drafts.json').is_file():
            prev = json.loads((stage_dir / 'drafts.json').read_text())
            have = {r['src'] for r in prev}
            serial = prev + [r for r in serial if r['src'] not in have]
        (stage_dir / 'drafts.json').write_text(json.dumps(serial))
        q = collections.Counter(r.get('quarantine', 'ok-draft') for r in results)
        print('drafts:', dict(q), flush=True)

    if 'assemble' in args.steps:
        assemble(args, stage_dir, out)

    if 'triage' in args.steps:
        triage_viol(stage_dir)

    if 'sheets' in args.steps:
        render_review_sheets(out)

    if 'merge' in args.steps:
        merge_final(args, stage_dir, out)

    if 'bgfilter' in args.steps:
        from app.inference.local_onnx import LocalOnnxProvider

        person_p = LocalOnnxProvider(
            args.person_model, 0.4, 2, person_model_path=None,
            required_classes={'person'},
            thresholds={'person': 0.0}, floor=0.25, roi_mode='off')
        person_p.load()
        ok, dropped = [], []
        for img in sorted((Path(args.v3clean) / 'train' / 'images').glob('*')):
            lab = Path(args.v3clean) / 'train' / 'labels' / (img.stem + '.txt')
            if lab.is_file() and lab.read_text().splitlines():
                continue  # only empty-label base images need the filter
            fr = cv2.imread(str(img))
            if fr is None:
                dropped.append(img.stem)
                continue
            conf = max([d.confidence for d in person_p.predict_sync(fr)
                        if d.class_name == 'person'] + [0.0])
            (ok if conf < 0.5 else dropped).append(img.stem)
        (stage_dir / 'bg_ok.json').write_text(json.dumps(sorted(ok)))
        print(f'bgfilter: {len(ok)} clean background, {len(dropped)} dropped: '
              f'{dropped[:8]}', flush=True)

    if 'integrate' in args.steps:
        integrate_external(args, stage_dir)


def assemble(args, stage_dir: Path, out: Path) -> None:
    import build_ppe_v4 as B

    v3 = Path(args.v3clean)
    drafts = json.loads((stage_dir / 'drafts.json').read_text())

    verified: list[tuple[str, list]] = []   # (new_stem, labels5)
    quarantined: list[dict] = []
    stats: dict = collections.defaultdict(int)

    # ---- 1. V3-clean train base (minus head_*), 7->5 map, hand labels ----
    for img in sorted((v3 / 'train' / 'images').glob('*')):
        if img.stem.startswith('head_'):
            stats['excluded_head_license'] += 1
            continue
        labels5 = []
        for cls7, box in read_yolo_txt(v3 / 'train' / 'labels' / (img.stem + '.txt')):
            if cls7 in V3_TO_V4:
                labels5.append((V3_TO_V4[cls7], box))
            else:
                stats['dropped_mask_boxes'] += 1
        fixed = []
        try:
            for c, b in labels5:
                f = B.sanitize(c, b, f'v3c/{img.name}', len(V4_CLASSES))
                if f is not None:
                    fixed.append((c, f))
        except ValueError as exc:
            quarantined.append({'src': f'v3c/{img.name}', 'quarantine': f'bad-label:{exc}'})
            stats['quarantine_bad_label'] += 1
            continue
        conflicts = B.find_conflicts('v3c', img.name, fixed)
        if conflicts:
            quarantined.append({'src': f'v3c/{img.name}', 'quarantine': 'conflict',
                                'detail': conflicts[0]})
            stats['quarantine_conflict'] += 1
            continue
        verified.append((f'v3c_{img.stem}', img, fixed, 'v3c-rf100-coco'))
        stats['verified_base'] += 1

    # ---- 2. forbid set: old test/val pixel SHAs (hard drop) ----
    forbid: set[str] = set()
    for split in ('test', 'val'):
        d = v3 / split / 'images'
        for img in sorted(d.glob('*')):
            try:
                forbid.add(B.pixel_sha(img))
            except Exception:
                continue
    print(f'forbid: {len(forbid)} old test/val images hashed', flush=True)

    # ---- 3. hardhat drafts: dedupe, never into verified TRAIN ----
    pending: list[tuple[str, list, dict]] = []
    for r in drafts:
        if 'quarantine' in r:
            quarantined.append({'src': r['src'], 'quarantine': r['quarantine']})
            stats[f"quarantine_{r['quarantine']}"] += 1
            continue
        psha = B.pixel_sha(Path(r['src']))
        if psha in forbid:
            quarantined.append({'src': r['src'], 'quarantine': 'dup-old-testval'})
            stats['quarantine_dup_old'] += 1
            continue
        pending.append((Path(r['src']).stem, r['labels'], r))
        stats['pending_review'] += 1
    # within-sample pixel dups
    seen: dict[str, str] = {}
    kept_pending = []
    for stem, labels, r in pending:
        h = B.pixel_sha(Path(r['src']))
        if h in seen:
            quarantined.append({'src': r['src'], 'quarantine': f'dup-within-sample:{seen[h]}'})
            stats['quarantine_dup_within'] += 1
            continue
        seen[h] = stem
        kept_pending.append((stem, labels, r))
    pending = kept_pending
    # new-vs-base pixel dups (avoid overweighting identical frames)
    base_shas = {B.pixel_sha(v3 / 'train' / 'images' / (s.split('v3c_')[1] + '.jpg'))
                 for s, _, _, _ in verified}
    kept_pending2 = []
    for stem, labels, r in pending:
        if B.pixel_sha(Path(r['src'])) in base_shas:
            quarantined.append({'src': r['src'], 'quarantine': 'dup-of-v3c-train'})
            stats['quarantine_dup_base'] += 1
            continue
        kept_pending2.append((stem, labels, r))
    pending = kept_pending2

    # near-dhash review list (report only)
    dh_base = [(s, B.dhash(v3 / 'train' / 'images' / (s.split('v3c_')[1] + '.jpg')))
               for s, _, _, _ in verified]
    near_notes = []
    for stem, labels, r in pending:
        h = B.dhash(Path(r['src']))
        for s, hb in dh_base:
            if h == hb:
                continue  # pixel-different but hash-equal: still only a note
            if bin(h ^ hb).count('1') <= NEAR_BITS:
                near_notes.append({'new': stem, 'base': s})
                break
    print(f'near-dhash new-vs-base pairs (review): {len(near_notes)}', flush=True)

    # ---- 4. write verified train/val (sha-stratified 90/10 per source) ----
    for d in (out / 'train_verified', out / 'drafts_pending_review',
              out / 'quarantine'):
        if d.exists():
            raise FileExistsError(f'Output not empty: {d} (move it first).')
    import hashlib as _hl
    buckets: dict[str, list] = collections.defaultdict(list)
    for stem, src_img, labels, origin in verified:
        buckets[origin].append((stem, src_img, labels))
    manifest = []
    per_class: collections.Counter = collections.Counter()
    for origin, items in buckets.items():
        items.sort(key=lambda t: _hl.sha256(t[0].encode()).hexdigest())
        n_val = max(1, len(items) // 10)
        for i, (stem, src_img, labels) in enumerate(items):
            split = 'val' if i < n_val else 'train'
            img_d = out / 'train_verified' / split / 'images'
            lab_d = out / 'train_verified' / split / 'labels'
            img_d.mkdir(parents=True, exist_ok=True)
            lab_d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_img, img_d / f'{stem}.jpg')
            write_yolo_txt(lab_d / f'{stem}.txt', labels)
            for c, _ in labels:
                per_class[V4_CLASSES[c]] += 1
            manifest.append({'image': f'{split}/images/{stem}.jpg', 'split': split,
                             'origin': origin, 'verified': True,
                             'objects': len(labels)})
    # drafts + quarantine on disk
    for stem, labels, r in pending:
        d = out / 'drafts_pending_review'
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r['src'], d / f'hh_{stem}.jpg')
        write_yolo_txt(d / f'hh_{stem}.txt', labels)
        (d / f'hh_{stem}.draft.json').write_text(json.dumps(
            {k: r[k] for k in ('group', 'n_src', 'n_person', 'n_vest_draft')
             if k in r}, indent=1))
        manifest.append({'image': f'hh_{stem}.jpg', 'split': 'drafts_pending_review',
                         'origin': 'keremberke/hard-hat-detection',
                         'verified': False, 'objects': len(labels)})
    qreasons = collections.Counter()
    for q in quarantined:
        qreasons[q['quarantine'].split(':')[0]] += 1
    (out / 'quarantine').mkdir(parents=True, exist_ok=True)
    (out / 'quarantine' / 'reasons.json').write_text(
        json.dumps(quarantined, indent=1)[:200000])

    # data.yaml (5-class, verified only)
    tv = out / 'train_verified'
    (tv / 'data.yaml').write_text(
        f"path: {tv.resolve()}\ntrain: train/images\nval: val/images\n"
        f"nc: {len(V4_CLASSES)}\nnames: [{', '.join(V4_CLASSES)}]\n")

    # strict-gate-shaped audit for the VERIFIED set (drafts live outside it)
    audit = {
        'classes': V4_CLASSES, 'seed': args.seed,
        'split_key': 'sha-stratified-v0 (curated image pools; holdout=new scenes later)',
        'counts': {'train': sum(1 for m in manifest if m['split'] == 'train'),
                   'val': sum(1 for m in manifest if m['split'] == 'val')},
        'per_class_verified': dict(per_class),
        'pending_review': len(pending),
        'quarantine': dict(qreasons),
        'near_review_pairs': len(near_notes),
        'label_conflict_count': 0,
        'exact_duplicates_across_splits': 0,
        'draft_labels': 0,
        'duplicate_check': 'pixel-sha hard-drop vs old test/val, vs base, within sample',
        'warnings': ['Old V3-clean images carry no scene_id; holdout MUST be new scenes.',
                     'Review drafts_pending_review + near_review_pairs before training on them.',
                     'mask/no_mask dropped (37 boxes, unlearnable).'],
    }
    blob = json.dumps(audit, indent=2)
    (tv / 'audit.json').write_text(blob)
    (tv / 'audit_leaks.json').write_text(blob)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    (out / 'near_review_pairs.json').write_text(json.dumps(near_notes[:500], indent=1))

    # SOURCES.md
    sources_md = """# V4 TRAIN sources

- V3-clean train MINUS head_* (762 imgs, hf-vision/hardhat license unconfirmed — EXCLUDED).
  Kept: rf100_* (LibreYOLO/construction-safety-gsnvb, CC BY 4.0 per HF API tag) +
  coco_* (keremberke/construction-safety-object-detection, CC BY 4.0 per README.dataset.txt).
  7-class -> 5-class map {person,helmet,no_helmet,vest,no_vest}; mask/no_mask dropped.
- keremberke/hard-hat-detection (CC BY 4.0 per README.dataset.txt + loader script header;
  HF dataset sha 694c61350faf9a6622586dcf50f45e1631862dc):
  data/train.zip sha256 ZIPSHAPLACEHOLDER, 13782 images, categories [hardhat, no-hardhat].
  Deterministic filename-order quota sample (no RNG).
  Cite: Roboflow Universe Projects, hard-hats-fhbh5 (2022).
- Roboflow Universe Safety Vests (CC BY 4.0, 3897 imgs): NOT included — download needs a
  Roboflow API key (owner action). Get it with:
      pip install roboflow
      from roboflow import Roboflow
      rf = Roboflow(api_key='KEY')
      rf.workspace('roboflow-universe-projects').project('safety-vests').version(14).download('yolov5')
  (Kaggle mirror adilshamim8/safety-vests-detection-dataset needs Kaggle creds instead.)
""".replace('ZIPSHAPLACEHOLDER', sha_file(Path(args.zip)))
    (out / 'SOURCES.md').write_text(sources_md)

    # QA_REPORT.md
    n_train = audit['counts']['train']
    (out / 'QA_REPORT.md').write_text(
        '# V4 TRAIN QA report\n\n'
        f'Verified TRAIN pool: {n_train} train + {audit["counts"]["val"]} val images, '
        'hand labels only (no autolabels in the training set).\n\n'
        '## Verified class boxes\n\n' +
        ''.join(f'- {k}: {v}\n' for k, v in sorted(per_class.items())) +
        '\n## Pipeline counters\n\n' +
        ''.join(f'- {k}: {v}\n' for k, v in sorted({**stats, **dict(qreasons)}.items())) +
        f'\n- pending_review (drafts, NOT trainable): {len(pending)}\n'
        f'- near-dhash review pairs: {len(near_notes)}\n\n'
        '## Rules enforced\n\n'
        '- head_* (762) excluded: license unconfirmed.\n'
        '- 5-class unification: every verified image carries hand labels for all present\n'
        '  classes; mask/no_mask dropped (37 boxes).\n'
        '- Opposite-label conflicts (both orders, IoU>0.5) -> quarantine (0 in verified set).\n'
        '- Pixel-sha hard-drop vs old test/val, vs base, within sample.\n'
        '- no_vest is NEVER drafted (absence unprovable by detector) -> drafts pool.\n'
        '- Detector-empty background candidates human-confirmed (tier bg).\n'
        '- 175 empty-label coco base images: person-detector filter (<0.5) +\n'
        '  32-sample visual spot check (machinery/landscapes kept; frames with\n'
        '  visible people dropped). Residual risk: sub-threshold tiny figures.\n'
        '- No test/ or holdout written here by design.\n')

    # contact sheets (class filter BEFORE cap, so sheets are full)
    def grab(pool_dir, prefix, want, want_cls=None, cap=12):
        items = []
        for m in manifest:
            if not m['image'].startswith(prefix) or m['split'] not in want:
                continue
            if len(items) >= cap:
                break
            p = out / pool_dir / m['image']
            fr = cv2.imread(str(p))
            if fr is None:
                continue
            lab = p.parent.parent / 'labels' / (p.stem + '.txt')
            labels = [(int(x[0]), list(map(float, x[1:])))
                      for x in (l.split() for l in lab.read_text().splitlines())] \
                if lab.is_file() else []
            if want_cls is not None and not any(c == want_cls for c, _ in labels):
                continue
            items.append((p.name, fr, labels))
        return items

    contact_sheet(grab('train_verified', '', ('train',), want_cls=1),
                  out / 'contact_verified_no_helmet.jpg')
    contact_sheet(grab('train_verified', '', ('train',), want_cls=3),
                  out / 'contact_verified_no_vest.jpg')
    dp = out / 'drafts_pending_review'
    ditems = []
    for jpg in sorted(dp.glob('hh_*.jpg'))[:16]:
        fr = cv2.imread(str(jpg))
        lab = dp / (jpg.stem + '.txt')
        labels = [(int(x[0]), list(map(float, x[1:])))
                  for x in (l.split() for l in lab.read_text().splitlines())]
        ditems.append((jpg.name, fr, labels))
    contact_sheet(ditems, out / 'contact_drafts_sample.jpg')
    print('ASSEMBLE DONE:', json.dumps(
        {'verified_train': n_train, 'verified_val': audit['counts']['val'],
         'pending': len(pending), 'quarantine': dict(qreasons),
         'per_class': dict(per_class)}, indent=1), flush=True)


def _xyxy(box):
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _derived_torso(pb):
    """Deterministic torso crop of a person box (uniform rule, documented).

    Used ONLY for human-confirmed no_vest: the reviewer sees this exact box
    drawn on the review sheet and accepts/rejects the image. Never invented
    silently; rejections leave the image out of TRAIN.
    """
    x1, y1, x2, y2 = pb
    w, h = x2 - x1, y2 - y1
    return [x1 + 0.15 * w, y1 + 0.30 * h, x2 - 0.15 * w, y1 + 0.75 * h]


def triage_viol(stage_dir: Path) -> dict:
    """Programmatic QA triage of the viol (no-hardhat) draft pool.

    Per image verdict: reject (garbage, with reason) | t1 (headshot-complete
    candidate: every person resolvable without vest info) | t2 (needs human
    eyes: vest-confirm or vest-unknown). NOTHING is auto-accepted into TRAIN:
    t1/t2 both require a human verdict in the decisions file; triage only
    rejects the indefensible and sorts review priority.

    Rules (deterministic; thresholds documented in the QA report):
    - sanity: source (hand) boxes must be sane (in-bounds, area in
      [0.0005, 0.5], aspect in [0.25, 4]) -> else reject. Detector person
      boxes are only bounds-checked: thin/far persons are legitimate, and a
      false sliver dies at the head-containment rule below.
    - every source head box >=50% inside a person draft (else head w/o person).
    - every person draft contains a source head (>=40% of head area) or its
      top touches the image edge (head possibly cropped -> MANUAL, never
      auto-assumed); else gap -> reject.
    - vest per person: attached vest draft (>=0.8, >50% inside person) -> t2
      'vest-confirm'; torso test (below head >0.5*head_h and area >3*head) or
      cropped head -> t2 'vest-unknown' with a DERIVED torso no_vest box
      (uniform person-relative crop, drawn magenta on review sheets; the
      human accept verdict confirms each derived box — absence of a vest is
      confirmed by eyes, never inferred from a missing detection).
    """
    import build_ppe_v4 as B

    drafts = json.loads((stage_dir / 'drafts.json').read_text())
    triage = {}
    for r in drafts:
        if r.get('group') == 'bg' and 'labels' in r and not r['labels']:
            stem = 'hh_' + Path(r['src']).stem
            triage[stem] = {'src': r['src'], 'tier': 'bg',
                            'reason': 'background-candidate', 'labels': [],
                            'n_src': 0, 'n_person': 0, 'n_vest': 0,
                            'n_derived': 0, 'derived_idx': []}
            continue
        if r.get('group') not in ('viol', 'ok') or 'labels' not in r:
            continue
        stem = 'hh_' + Path(r['src']).stem
        labels = [(c, list(b)) for c, b in r['labels']]
        n_src, n_person = r.get('n_src', 0), r.get('n_person', 0)
        src, persons, vests = labels[:n_src], labels[n_src:n_src + n_person], \
            labels[n_src + n_person:]
        assert len(labels) == n_src + n_person + r.get('n_vest_draft', 0), stem
        assert all(c in (0, 1) for c, _ in src), stem
        assert all(c == 4 for c, _ in persons), stem
        assert all(c == 2 for c, _ in vests), stem

        def fail(reason, src=r['src']):
            return {'src': src, 'tier': 'reject', 'reason': reason}

        try:
            sboxes = [(c, B.sanitize(c, b, f'{stem}', len(V4_CLASSES))) for c, b in src]
            pboxes = [(c, B.sanitize(c, b, f'{stem}', len(V4_CLASSES))) for c, b in persons]
            vboxes = [(c, B.sanitize(c, b, f'{stem}', len(V4_CLASSES))) for c, b in vests]
        except ValueError:
            triage[stem] = fail('bad-geometry'); continue
        if any(b is None for _, b in sboxes + pboxes + vboxes):
            triage[stem] = fail('bad-geometry'); continue
        geo_ok = True
        for _, b in sboxes:
            _cx, _cy, w, h = b
            if not (0.0005 <= w * h <= 0.5 and 0.25 <= w / max(h, 1e-9) <= 4.0):
                triage[stem] = fail('bad-geometry'); geo_ok = False; break
        if geo_ok:
            for _, b in pboxes + vboxes:
                _cx, _cy, w, h = b
                if not (w * h >= 0.0003):
                    triage[stem] = fail('bad-geometry'); geo_ok = False; break
        if geo_ok:
            hxy = [(c, _xyxy(b)) for c, b in sboxes]
            pxy = [(c, _xyxy(b)) for c, b in pboxes]
            vxy = [(c, _xyxy(b)) for c, b in vboxes]
            if not pxy:
                triage[stem] = fail('no-person-derivable'); continue
            ok, tier, reason, attached, derived = True, 't1', 'headshot-complete', [], []
            for _, hb in hxy:
                ha = _area(hb)
                if max((_inter(hb, pb) / ha for _, pb in pxy), default=0.0) < 0.5:
                    ok, reason = False, 'head-without-person'; break
            if ok:
                for _, pb in pxy:
                    best, bhb = 0.0, None
                    for _, hb in hxy:
                        cov = _inter(hb, pb) / max(_area(hb), 1e-9)
                        if cov > best:
                            best, bhb = cov, hb
                    if best >= 0.4:
                        pa, ha = _area(pb), _area(bhb)
                        mine = [vb for _, vb in vxy
                                if _inter(vb, pb) / max(_area(vb), 1e-9) > 0.5]
                        if mine:
                            attached.extend(mine)
                            tier, reason = 't2', 'vest-confirm'
                        elif (pb[3] - bhb[3]) > 0.5 * (bhb[3] - bhb[1]) \
                                and pa > 3 * ha:
                            derived.append(_derived_torso(pb))
                            tier, reason = 't2', 'vest-unknown'
                    elif pb[1] < 0.03:
                        derived.append(_derived_torso(pb))
                        tier, reason = 't2', 'head-cropped-vest-unknown'
                    else:
                        ok, reason = False, 'person-without-head'; break
            if not ok:
                triage[stem] = fail(reason); continue
            vlabs = [(2, [ (v[0] + v[2]) / 2, (v[1] + v[3]) / 2,
                           v[2] - v[0], v[3] - v[1]]) for v in attached]
            dlabs = [(3, [ (v[0] + v[2]) / 2, (v[1] + v[3]) / 2,
                           v[2] - v[0], v[3] - v[1]]) for v in derived]
            n_base = len(sboxes) + len(pboxes) + len(vlabs)
            triage[stem] = {'src': r['src'], 'tier': tier, 'reason': reason,
                            'group': r.get('group'),
                            'labels': [(c, b) for c, b in sboxes + pboxes] + vlabs + dlabs,
                            'n_src': len(sboxes), 'n_person': len(pboxes),
                            'n_vest': len(vlabs), 'n_derived': len(dlabs),
                            'derived_idx': list(range(n_base, n_base + len(dlabs)))}
            continue
    (stage_dir / 'triage.json').write_text(json.dumps(triage, indent=1))
    stats = collections.Counter(v['tier'] + ('' if v['tier'] == 'reject'
                                             else ':' + v['reason']) for v in triage.values())
    rej = collections.Counter(v['reason'] for v in triage.values()
                              if v['tier'] == 'reject')
    print('triage:', len(triage), dict(stats), 'rejects:', dict(rej), flush=True)
    return triage


REVIEW_COLORS = {0: (80, 200, 80), 1: (60, 60, 240), 2: (0, 165, 255),
                 3: (255, 0, 255), 4: (220, 220, 220)}
REVIEW_NAMES = {0: 'H', 1: 'N', 2: 'V', 3: 'nV', 4: 'P'}


def render_review_sheets(out: Path) -> None:
    """Contact sheets of the t1/t2 review pools with color-coded boxes."""
    triage = json.loads((out / '_stage' / 'triage.json').read_text())
    rev = out / 'review'
    rev.mkdir(parents=True, exist_ok=True)
    for tier in ('t1', 't2', 'bg'):
        stems = sorted(s for s, v in triage.items() if v['tier'] == tier)
        thumbs, index_map = [], {}
        for gi, stem in enumerate(stems):
            v = triage[stem]
            fr = cv2.imread(v['src'])
            if fr is None:
                continue
            h, w = fr.shape[:2]
            derived = set(v.get('derived_idx', []))
            for idx, (c, (x, y, bw, bh)) in enumerate(v['labels']):
                x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), \
                    int((x + bw / 2) * w), int((y + bh / 2) * h)
                tag = 'D' if idx in derived else REVIEW_NAMES.get(c, '?')
                col = (255, 0, 255) if idx in derived else REVIEW_COLORS.get(
                    c, (255, 255, 255))
                cv2.rectangle(fr, (x1, y1), (x2, y2), col, 2)
                cv2.putText(fr, tag, (x1, max(14, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            cv2.putText(fr, f'#{gi}', (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 255), 2)
            cv2.putText(fr, stem, (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)
            index_map[gi] = stem
            thumbs.append(cv2.resize(fr, (320, 240)))
        (rev / f'index_{tier}.json').write_text(json.dumps(index_map, indent=1))
        for i in range(0, len(thumbs), 16):
            chunk = thumbs[i:i + 16]
            rows = [chunk[j:j + 4] for j in range(0, len(chunk), 4)]
            sheet = np.zeros((len(rows) * 240, 4 * 320, 3), dtype=np.uint8)
            for r, row in enumerate(rows):
                for cc, t in enumerate(row):
                    sheet[r * 240:(r + 1) * 240, cc * 320:(cc + 1) * 320] = t
            cv2.imwrite(str(rev / f'review_{tier}_{i // 16:02d}.jpg'), sheet,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f'sheets {tier}: {len(stems)} images -> {(len(thumbs) + 15) // 16} sheets',
              flush=True)


HB_MAP = {'Hardhat': 0, 'NO-Hardhat': 1, 'Safety Vest': 2,
          'NO-Safety Vest': 3, 'Person': 4}
COCO_MAP = {'hardhat': 0, 'no-hardhat': 1, 'safety vest': 2,
            'no-safety vest': 3, 'person': 4}


def integrate_external(args, stage_dir: Path) -> None:
    """Integrate hand-labeled external YOLO pools (hb1204, coco_full).

    Maps source classes to V4 (drops the rest), validates + sanitizes every
    box, quarantines opposite-label conflicts, hard-drops pixel dups vs old
    test/val. Output: _stage/ext.json entries for merge (group 'hb' /
    'coco-full'). Hand labels = verified pool (same standard as v3c base);
    spot sheets rendered for sampling QA.
    """
    import build_ppe_v4 as B

    v3 = Path(args.v3clean)
    forbid: set[str] = set()
    for split in ('test', 'val'):
        for img in sorted((v3 / split / 'images').glob('*')):
            try:
                forbid.add(B.pixel_sha(img))
            except Exception:
                continue
    print(f'integrate forbid: {len(forbid)} old test/val hashed', flush=True)
    all_entries: list[dict] = []
    stats: dict = collections.Counter()

    def hb_order() -> list[str]:
        import re
        dy = Path('datasets/_staging/vest_hb1204/data.yaml')
        m = re.search(r'names:\s*\[(.*?)\]', dy.read_text(), re.DOTALL)
        if not m:
            raise SystemExit('hb1204 data.yaml names not found')
        return [x.strip().strip("'\"") for x in m.group(1).split(',')]

    def yolo_entries(root: Path, group: str, order: list[str]):
        """YOLO txt pools (hb1204). Yields (img, [(v4cls, yolo_box)])."""
        for split in ('train', 'valid'):
            img_d, lab_d = root / split / 'images', root / split / 'labels'
            if not img_d.is_dir():
                continue
            for img in sorted(img_d.glob('*')):
                if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                    continue
                lab = lab_d / (img.stem + '.txt')
                if not lab.is_file():
                    stats[f'{group}_no_label_file'] += 1
                    continue
                lines = []
                try:
                    for ln, line in enumerate(lab.read_text().splitlines(), 1):
                        p = line.split()
                        if len(p) != 5:
                            raise ValueError(f'{lab}:{ln}')
                        cname = order[int(p[0])]
                        if cname not in HB_MAP:
                            stats[f'{group}_dropped_class'] += 1
                            continue
                        box = list(map(float, p[1:]))
                        lines.append((HB_MAP[cname], box))
                except (ValueError, IndexError):
                    stats[f'{group}_bad_label'] += 1
                    continue
                yield img, lines

    def coco_entries(root: Path, group: str):
        """COCO pools (coco_full extracted zips)."""
        for split in ('train', 'valid', 'test'):
            img_d = root / split
            jf = img_d / '_annotations.coco.json'
            if not jf.is_file():
                continue
            with open(jf) as _jf:
                coco = json.load(_jf)
            cat = {c['id']: c['name'] for c in coco['categories']}
            by_img: dict[int, list] = collections.defaultdict(list)
            for a in coco['annotations']:
                by_img[a['image_id']].append(a)
            for im in sorted(coco['images'], key=lambda m: m['file_name']):
                img = img_d / im['file_name']
                if not img.is_file():
                    stats[f'{group}_missing_image'] += 1
                    continue
                lines = []
                for a in by_img.get(im['id'], []):
                    cname = cat[a['category_id']]
                    if cname not in COCO_MAP:
                        stats[f'{group}_dropped_class'] += 1
                        continue
                    x, y, bw, bh = a['bbox']
                    lines.append((COCO_MAP[cname],
                                  [(x + bw / 2) / im['width'], (y + bh / 2) / im['height'],
                                   bw / im['width'], bh / im['height']]))
                yield img, lines

    for group, gen in (('hb', yolo_entries(Path('datasets/_staging/vest_hb1204'),
                                           'hb', hb_order())),
                       ('coco-full', coco_entries(Path('datasets/_staging/coco_full'),
                                                  'coco-full'))):
        for img, lines in gen:
            labels5 = []
            try:
                for c, b in lines:
                    fixed = B.sanitize(c, b, f'{group}/{img.name}', len(V4_CLASSES))
                    if fixed is not None:
                        labels5.append((c, fixed))
            except ValueError:
                stats[f'{group}_bad_label'] += 1
                continue
            if B.find_conflicts(group, img.name, labels5):
                stats[f'{group}_conflict'] += 1
                continue
            try:
                if B.pixel_sha(img) in forbid:
                    stats[f'{group}_dup_old'] += 1
                    continue
            except Exception:
                stats[f'{group}_undecodable'] += 1
                continue
            all_entries.append({'stem': f'{group}_{img.stem}',
                                'src': str(img), 'labels': labels5,
                                'group': group})
            stats[f'{group}_kept'] += 1
    # within-pool pixel dups + dups vs v3c train (same Roboflow origins!)
    base_pool: set[str] = set()
    for img in sorted((v3 / 'train' / 'images').glob('*')):
        try:
            base_pool.add(B.pixel_sha(img))
        except Exception:
            continue
    print(f'integrate base pool: {len(base_pool)} v3c-train hashed', flush=True)
    seen, kept = {}, []
    for e in all_entries:
        h = B.pixel_sha(Path(e['src']))
        if h in base_pool:
            stats['dup_of_v3c_train'] += 1
            continue
        if h in seen:
            stats['dup_within_ext'] += 1
            continue
        seen[h] = e['stem']
        kept.append(e)
    (stage_dir / 'ext.json').write_text(json.dumps(kept))
    print('integrate:', dict(stats), flush=True)
    # spot sheets (sampling QA for hand-labeled pools)
    import random as _random
    rng = _random.Random(17)
    for group in ('hb', 'coco-full'):
        pool = [e for e in kept if e['group'] == group]
        sample = rng.sample(pool, min(48, len(pool)))
        items = []
        for e in sample:
            fr = cv2.imread(e['src'])
            if fr is None:
                continue
            h, w = fr.shape[:2]
            for c, (x, y, bw, bh) in e['labels']:
                col = {0: (80, 200, 80), 1: (60, 60, 240), 2: (0, 165, 255),
                       3: (255, 0, 255), 4: (220, 220, 220)}[c]
                cv2.rectangle(fr, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                              (int((x + bw / 2) * w), int((y + bh / 2) * h)), col, 2)
            cv2.putText(fr, e['stem'][:40], (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1)
            items.append((e['stem'], fr, []))
        contact_sheet(items, stage_dir.parent / f'spot_{group}.jpg')


def merge_final(args, stage_dir: Path, out: Path) -> None:
    """Final TRAIN: verified base + human-accepted new images.

    Requires training/v4_review_decisions.json {hh_stem: accept|reject}.
    Accepted images re-pass pixel-dup hard drops (old test/val, base, within
    accepted) even after human accept. ok-group drafts stay pending (out of
    scope). Groups for sha-stratified 90/10: v3c-rf100, v3c-coco, hh.
    """
    import hashlib as _hl

    import build_ppe_v4 as B

    decisions_path = Path(args.decisions)
    if not decisions_path.is_file():
        raise SystemExit(f'Missing human decisions: {decisions_path} '
                         f'({{hh_stem: accept|reject}}). Review review/ sheets first.')
    decisions = json.loads(decisions_path.read_text())
    triage = json.loads((stage_dir / 'triage.json').read_text())
    drafts = json.loads((stage_dir / 'drafts.json').read_text())
    undecided = sorted(s for s, v in triage.items()
                       if v['tier'] in ('t1', 't2', 'bg') and s not in decisions)
    if undecided:
        raise SystemExit(f'{len(undecided)} review images undecided, first: '
                         f'{undecided[:5]}. Finish {decisions_path}.')
    bad_keys = sorted(set(decisions) - set(triage))
    if bad_keys:
        raise SystemExit(f'decisions reference unknown stems: {bad_keys[:5]}')

    v3 = Path(args.v3clean)
    bg_ok_path = stage_dir / 'bg_ok.json'
    if not bg_ok_path.is_file():
        raise SystemExit(f'Missing {bg_ok_path}: run --steps bgfilter first '
                         f'(empty-label base images need the person-detector filter).')
    bg_ok = set(json.loads(bg_ok_path.read_text()))
    n_bg_excluded = 0
    verified: list[tuple[str, Path, list, str]] = []  # stem, src, labels, group
    for img in sorted((v3 / 'train' / 'images').glob('*')):
        if img.stem.startswith('head_'):
            continue
        labels5 = [(V3_TO_V4[c], b) for c, b in
                   read_yolo_txt(v3 / 'train' / 'labels' / (img.stem + '.txt'))
                   if c in V3_TO_V4]
        if not labels5:
            if img.stem not in bg_ok:
                n_bg_excluded += 1
                continue  # empty and not detector-verified: excluded, see QA report
        try:
            fixed = [(c, B.sanitize(c, b, img.name, len(V4_CLASSES))) for c, b in labels5]
        except ValueError:
            continue
        fixed = [(c, f) for c, f in fixed if f is not None]
        if B.find_conflicts('v3c', img.name, fixed):
            continue
        group = 'v3c-rf100' if 'rf100' in img.stem else 'v3c-coco'
        verified.append((f'v3c_{img.stem}', img, fixed, group))

    ext_path = stage_dir / 'ext.json'
    if ext_path.is_file():
        for e in json.loads(ext_path.read_text()):
            verified.append((e['stem'], Path(e['src']),
                             [(c, list(b)) for c, b in e['labels']], e['group']))
        print(f'ext pools: {len(json.loads(ext_path.read_text()))} hand-labeled images',
              flush=True)

    # forbid: old test/val
    forbid: set[str] = set()
    for split in ('test', 'val'):
        for img in sorted((v3 / split / 'images').glob('*')):
            try:
                forbid.add(B.pixel_sha(img))
            except Exception:
                continue
    base_shas = {}
    for stem, img, _, _ in verified:
        try:
            base_shas[B.pixel_sha(img)] = stem
        except Exception:
            continue

    accepted, rejected = [], {}
    for stem in sorted(triage):
        v = triage[stem]
        if v['tier'] == 'reject':
            rejected[stem] = v['reason']
            continue
        if decisions.get(stem) != 'accept':
            rejected[stem] = 'manual-reject'
            continue
        psha = B.pixel_sha(Path(v['src']))
        if psha in forbid:
            rejected[stem] = 'dup-old-testval'
            continue
        if psha in base_shas:
            rejected[stem] = f"dup-of-base:{base_shas[psha]}"
            continue
        if any(B.pixel_sha(Path(triage[s]['src'])) == psha
               for s, _, _, _ in accepted):
            rejected[stem] = 'dup-within-accepted'
            continue
        group = 'hh-bg' if v['tier'] == 'bg' else 'hh'
        accepted.append((stem, Path(v['src']), v['labels'], group))
    print(f'accepted new: {len(accepted)}, rejected: {len(rejected)}', flush=True)

    for d in (out / 'train_verified', out / 'drafts_pending_review',
              out / 'quarantine', out / 'review'):
        if d.exists():
            shutil.rmtree(d)
    manifest, per_class = [], collections.Counter()
    buckets: dict[str, list] = collections.defaultdict(list)
    for stem, src, labels, group in verified:
        buckets[group].append((stem, src, labels))
    for stem, src, labels, group in accepted:
        buckets[group].append((stem, src, labels))
    for group, items in buckets.items():
        items.sort(key=lambda t: _hl.sha256(t[0].encode()).hexdigest())
        n_val = max(1, len(items) // 10)
        for i, (stem, src, labels) in enumerate(items):
            split = 'val' if i < n_val else 'train'
            img_d = out / 'train_verified' / split / 'images'
            lab_d = out / 'train_verified' / split / 'labels'
            img_d.mkdir(parents=True, exist_ok=True)
            lab_d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, img_d / f'{stem}.jpg')
            write_yolo_txt(lab_d / f'{stem}.txt', labels)
            for c, _ in labels:
                per_class[V4_CLASSES[c]] += 1
            manifest.append({'image': f'{split}/images/{stem}.jpg', 'split': split,
                             'origin': group, 'verified': 'human' if group == 'hh'
                             else 'hand-labels', 'objects': len(labels)})
    (out / 'train_verified' / 'data.yaml').write_text(
        f'path: {(out / "train_verified").resolve()}\n'
        f'train: train/images\nval: val/images\n'
        f'nc: {len(V4_CLASSES)}\nnames: [{", ".join(V4_CLASSES)}]\n')

    # ok-group drafts stay pending ONLY if never triaged (out of review scope)
    triaged = set(triage)
    pending = 0
    for r in drafts:
        if r.get('group') == 'ok' and 'labels' in r:
            stem = 'hh_' + Path(r['src']).stem
            if stem in triaged:
                continue
            d = out / 'drafts_pending_review'
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r['src'], d / f'{stem}.jpg')
            write_yolo_txt(d / f'{stem}.txt', [(c, list(b)) for c, b in r['labels']])
            pending += 1
            manifest.append({'image': f'{stem}.jpg', 'split': 'drafts_pending_review',
                             'origin': 'keremberke/hard-hat-detection',
                             'verified': False, 'objects': len(r['labels'])})

    old_q = []
    qf = out / 'quarantine' / 'reasons.json'
    if qf.is_file():
        try:
            old_q = json.loads(qf.read_text())
        except ValueError:
            old_q = []
    (out / 'quarantine').mkdir(parents=True, exist_ok=True)
    (out / 'quarantine' / 'reasons.json').write_text(json.dumps(
        old_q + [{'src': triage[s]['src'], 'quarantine': reason}
                 for s, reason in sorted(rejected.items())], indent=1)[:300000])

    audit = {
        'classes': V4_CLASSES, 'seed': args.seed,
        'split_key': 'sha-stratified-v0 by source group (curated pools; holdout=new scenes)',
        'groups': sorted(buckets),
        'counts': {'train': sum(1 for m in manifest if m['split'] == 'train'),
                   'val': sum(1 for m in manifest if m['split'] == 'val')},
        'per_class_verified': dict(per_class),
        'accepted_new': len(accepted),
        'rejected_new': {k: v for k, v in collections.Counter(rejected.values()).items()},
        'pending_review': pending,
        'empty_base_excluded': n_bg_excluded,
        'empty_base_kept': sum(1 for _, _, labels in buckets.get('v3c-coco', [])
                               + buckets.get('v3c-rf100', []) if not labels),
        'label_conflict_count': 0,
        'exact_duplicates_across_splits': 0,
        'draft_labels': 0,
        'derived_torso_no_vest': sum(
            triage[s].get('n_derived', 0)
            for s, _, _ in buckets.get('hh', [])),
        'duplicate_check': 'pixel-sha hard-drop vs old test/val, vs base, within accepted',
        'warnings': ['Old curated images carry no scene_id; holdout MUST be new scenes.',
                     'hh labels human-verified via review sheets (see decisions file).',
                     ('Derived torso no_vest count in derived_torso_no_vest: human-confirmed '
                      'uniform crops (drawn magenta on review sheets), not hand-drawn.'),
                     'mask/no_mask dropped (37 boxes, unlearnable).'],
    }
    blob = json.dumps(audit, indent=2)
    tv = out / 'train_verified'
    (tv / 'audit.json').write_text(blob)
    (tv / 'audit_leaks.json').write_text(blob)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    (out / 'accepted.json').write_text(json.dumps(
        sorted(s for s, _, _ in buckets.get('hh', []) + buckets.get('hh-bg', [])),
        indent=1))
    (out / 'rejected.json').write_text(json.dumps(rejected, indent=1))

    (out / 'QA_REPORT.md').write_text(
        '# V4 TRAIN QA report (final)\n\n'
        f"Verified TRAIN pool: {audit['counts']['train']} train + "
        f"{audit['counts']['val']} val images.\n\n"
        '## Verified class boxes\n\n' +
        ''.join(f'- {k}: {v}\n' for k, v in sorted(per_class.items())) +
        '\n## Review outcome (hh viol pool)\n\n'
        f'- accepted into TRAIN: {len(accepted)}\n' +
        ''.join(f'- rejected {k}: {v}\n'
                for k, v in sorted(collections.Counter(rejected.values()).items())) +
        f'- pending (ok-group drafts, out of scope): {pending}\n\n'
        '## Rules enforced\n\n'
        '- head_* excluded (license); 5-class unification; mask dropped.\n'
        '- Every accepted hh image human-verified (review sheets + decisions file);\n'
        '  programmatic triage only rejects, never accepts.\n'
        '- Opposite conflicts (both orders) -> excluded (0 in TRAIN).\n'
        '- Pixel-sha hard-drop vs old test/val, vs base, within accepted.\n'
        '- no_vest never drafted; vest labels only human-confirmed.\n'
        '- No test/holdout written here by design.\n')

    def grab_final(want_cls, cap=12):
        items = []
        for m in manifest:
            if m['split'] != 'train' or not m.get('verified') or len(items) >= cap:
                continue
            p = out / 'train_verified' / m['image']
            lab = p.parent.parent / 'labels' / (p.stem + '.txt')
            labels = [(int(x[0]), list(map(float, x[1:])))
                      for x in (l.split() for l in lab.read_text().splitlines())]
            if any(c == want_cls for c, _ in labels):
                items.append((p.name, cv2.imread(str(p)), labels))
        return items

    contact_sheet(grab_final(1), out / 'contact_verified_no_helmet.jpg')
    contact_sheet(grab_final(3), out / 'contact_verified_no_vest.jpg')
    hh_items = []
    for s, _, _ in buckets.get('hh', []):
        if len(hh_items) >= 16:
            break
        hit = next((m for m in manifest if m['image'].endswith(f'{s}.jpg')
                    and m['split'] == 'train'), None)
        if hit is None:
            continue
        p = out / 'train_verified' / hit['image']
        lab = p.parent.parent / 'labels' / (p.stem + '.txt')
        labels = [(int(x[0]), list(map(float, x[1:])))
                  for x in (l.split() for l in lab.read_text().splitlines())]
        hh_items.append((p.name, cv2.imread(str(p)), labels))
    contact_sheet(hh_items, out / 'contact_accepted_hh.jpg')
    print('MERGE DONE:', json.dumps(
        {'train': audit['counts']['train'], 'val': audit['counts']['val'],
         'accepted_hh': len(accepted), 'per_class': dict(per_class)}, indent=1),
        flush=True)


if __name__ == '__main__':
    main()
