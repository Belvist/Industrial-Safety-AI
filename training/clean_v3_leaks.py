"""Remove cross-split near-duplicate leakage from the auditor V3 dataset.

Keeps the 7-class format and all labels untouched. For every dHash cluster
spanning several splits, keeps ONE copy by priority test > val > train and
drops the twins (train loses a few dozen images out of 2069 — negligible).
Also runs the helmet/no_helmet + vest/no_vest overlap conflict audit.

Usage:
    /usr/bin/python3 training/clean_v3_leaks.py \
        --src ppe_public_v3 --out datasets/ppe_v3_clean --near-bits 8

Writes datasets/ppe_v3_clean/{train,val,test}, data.yaml, audit_leaks.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

from build_ppe_v3 import BKTree, dhash, iou_xyxy, pixel_sha  # noqa: E402


def load_split(src: Path, split: str):
    imgs = sorted((src / split / 'images').iterdir())
    out = []
    for img in imgs:
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        lab = src / split / 'labels' / f'{img.stem}.txt'
        labels = []
        if lab.is_file():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f'Bad label line in {lab}')
                c, vals = int(parts[0]), [float(v) for v in parts[1:]]
                x, y, w, h = vals
                labels.append((c, [x - w / 2, y - h / 2, x + w / 2, y + h / 2], vals))
        out.append({'split': split, 'image': img, 'labels': labels,
                    'dh': dhash(img), 'psha': pixel_sha(img)})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', default='ppe_public_v3')
    p.add_argument('--out', default='datasets/ppe_v3_clean')
    p.add_argument('--near-bits', type=int, default=8)
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    recs = load_split(src, 'train') + load_split(src, 'val') + load_split(src, 'test')
    print(f'loaded: {len(recs)} images')

    # Direct cross-split pairs only: NO transitive chains (single-linkage
    # chaining merges unrelated images through intermediate lookalikes).
    import itertools

    hashes = [r['dh'] for r in recs]
    pairs = []
    for a, b in itertools.combinations(range(len(recs)), 2):
        if recs[a]['split'] == recs[b]['split']:
            continue
        dist = bin(hashes[a] ^ hashes[b]).count('1')
        if dist <= args.near_bits or recs[a]['psha'] == recs[b]['psha']:
            pairs.append((dist, a, b))
    pairs.sort()
    prio = {'test': 0, 'val': 1, 'train': 2}
    keep = [True] * len(recs)
    leaks = []
    for dist, a, b in pairs:
        if not keep[a] or not keep[b]:
            continue
        drop, stay = (a, b) if prio[recs[a]['split']] > prio[recs[b]['split']] else (b, a)
        keep[drop] = False
        leaks.append({'kept': f"{recs[stay]['split']}/{recs[stay]['image'].name}",
                      'dropped': f"{recs[drop]['split']}/{recs[drop]['image'].name}",
                      'bits': dist})
    dropped = sum(1 for k in keep if not k)

    conflicts = []
    for i, r in enumerate(recs):
        if not keep[i]:
            continue
        boxes = [(c, b) for c, b, _ in r['labels']]
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                (c1, b1), (c2, b2) = boxes[a], boxes[b]
                if ((c1 in (1, 2) and c2 in (1, 2) and c1 != c2)
                        or (c1 in (3, 4) and c2 in (3, 4) and c1 != c2)):
                    if iou_xyxy(b1, b2) > 0.5:
                        conflicts.append({'image': r['image'].name, 'split': r['split'],
                                          'a': c1, 'b': c2})

    if out.exists():
        raise FileExistsError(f'Output not empty: {out} (move it first)')
    counts: dict[str, int] = collections.Counter()
    per_class: dict[int, int] = collections.Counter()
    for i, r in enumerate(recs):
        if not keep[i]:
            continue
        img_d = out / r['split'] / 'images'
        lab_d = out / r['split'] / 'labels'
        img_d.mkdir(parents=True, exist_ok=True)
        lab_d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(r['image'], img_d / r['image'].name)
        src_lab = src / r['split'] / 'labels' / f"{r['image'].stem}.txt"
        if src_lab.is_file():
            shutil.copyfile(src_lab, lab_d / f"{r['image'].stem}.txt")
        else:
            (lab_d / f"{r['image'].stem}.txt").write_text('')
        counts[r['split']] += 1
        for c, _, _ in r['labels']:
            per_class[c] += 1
    (out / 'data.yaml').write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: 7\nnames: [person, helmet, no_helmet, vest, no_vest, mask, no_mask]\n")
    report = {'input_images': len(recs),
              'cross_split_pairs': leaks, 'leak_pair_count': len(leaks),
              'dropped_twins': dropped,
              'counts': dict(counts), 'per_class': {str(k): v for k, v in per_class.items()},
              'conflicts': conflicts[:200], 'conflict_count': len(conflicts)}
    (out / 'audit_leaks.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({'pairs': len(leaks), 'dropped': dropped, 'counts': dict(counts),
                      'conflicts': len(conflicts)}, indent=2))


if __name__ == '__main__':
    main()
