"""Assemble the V3 violation-focused PPE dataset.

Sources: existing RF100 + COCO caches (see build_public_ppe) plus a Roboflow
Universe export of Personal Protective Equipment - Combined Model
(CC BY 4.0, 14 classes). Roboflow download needs a free API key and is done
separately (see --rb-dir); this script never touches the network.

Target: 4 PPE classes only (person stays on the dedicated person model):
    0 helmet, 1 no_helmet, 2 vest, 3 no_vest

Pipeline:
  1. Load YOLO-format labels from every source dir, map to target classes,
     sanitize boxes (clip rounding overflow, drop degenerate).
  2. Per-source / per-class stats (printed + audit.json).
  3. Exact-duplicate removal (pixel SHA256) AND near-duplicate clustering
     (dHash hamming <= --near-bits, union-find). Whole clusters stay in ONE
     split: no leakage of near-identical video frames across splits.
  4. Conflict audit: helmet/no_helmet or vest/no_vest boxes with IoU >= 0.5 on
     the same image are reported (not silently kept).
  5. Violation-focused sampling: images with no_helmet/no_vest first, then
     fill to --max-images, at least --min-per-class boxes per target class.
  6. Deterministic stratified split by scene cluster (seed fixed).

Roboflow export layout expected (YOLOv8 download, any version):
    <rb-dir>/train|valid|test/images + labels, data.yaml with 14 names.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageStat

CLASSES = ['helmet', 'no_helmet', 'vest', 'no_vest']

# Normalized Roboflow Combined names -> canonical target name.
RB_MAP = {
    'hardhat': 'helmet', 'helmet': 'helmet',
    'no-hardhat': 'no_helmet', 'no-helmet': 'no_helmet', 'no hardhat': 'no_helmet',
    'safety vest': 'vest', 'vest': 'vest', 'safety-vest': 'vest',
    'no-safety vest': 'no_vest', 'no-vest': 'no_vest', 'no vest': 'no_vest',
    'no-safety-vest': 'no_vest',
    'person': 'person',
}
# Canonical v1 RF100/COCO names usable as targets.
V1_CANON = {'helmet', 'no_helmet', 'vest', 'no_vest', 'mask', 'no_mask', 'person'}

EDGE_TOL = 0.005
NEAR_BITS_DEFAULT = 6
SPLITS = {'train': 'train', 'valid': 'val', 'test': 'test'}


def _norm(name: str) -> str:
    return ' '.join(name.lower().replace('_', '-').split())


def sanitize(cls: int, coords: list[float], where: str,
             n_classes: int | None = None) -> list[float] | None:
    bound = n_classes if n_classes is not None else len(CLASSES)
    if not 0 <= cls < bound:
        raise ValueError(f'{where}: invalid target class {cls}')
    if len(coords) != 4 or any(not -EDGE_TOL <= v <= 1 + EDGE_TOL for v in coords):
        raise ValueError(f'{where}: bbox out of range {coords}')
    x, y, w, h = coords
    if w <= 0 or h <= 0:
        raise ValueError(f'{where}: non-positive size {coords}')
    x1, y1, x2, y2 = max(0.0, x - w / 2), max(0.0, y - h / 2), min(1.0, x + w / 2), min(1.0, y + h / 2)
    if x1 >= x2 - 1e-6 or y1 >= y2 - 1e-6:
        return None
    if x2 - x1 > 1 + EDGE_TOL or y2 - y1 > 1 + EDGE_TOL:
        raise ValueError(f'{where}: invalid bbox geometry {coords}')
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def dhash(path: Path) -> int:
    with Image.open(path) as img:
        small = img.convert('L').resize((9, 8), Image.BILINEAR)
        px = list(small.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (1 if px[r * 9 + c] > px[r * 9 + c + 1] else 0)
    return bits


def pixel_sha(path: Path) -> str:
    with Image.open(path) as img:
        rgb = img.convert('RGB')
        return hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


@dataclass
class Record:
    source: str
    image: Path
    labels: list[tuple[int, list[float]]]
    dh: int = 0
    psha: str = ''


OPPOSITES = {('helmet', 'no_helmet'), ('vest', 'no_vest')}


def load_yolo_dir(root: Path, source: str, stats: dict, conflicts: list,
                  target: list[str] | None = None) -> list[Record]:
    target = list(target) if target else list(CLASSES)
    index = {name: i for i, name in enumerate(target)}
    names_file = root / 'data.yaml'
    if names_file.is_file():
        import yaml
        names = yaml.safe_load(names_file.read_text()).get('names', [])
        order = names if isinstance(names, list) else [names[str(i)] for i in range(len(names))]
    else:
        order = None
    records = []
    for src_split in ('train', 'valid', 'test'):
        imgs, labs = root / src_split / 'images', root / src_split / 'labels'
        if not imgs.is_dir():
            continue
        for img_file in sorted(imgs.iterdir()):
            if img_file.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            label_file = labs / f'{img_file.stem}.txt'
            labels = []
            for line in (label_file.read_text().splitlines() if label_file.is_file() else []):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f'Bad YOLO line in {label_file}')
                raw_cls, coords = int(parts[0]), [float(v) for v in parts[1:]]
                if order is None:
                    raise ValueError(f'{root}: data.yaml names required')
                canon = RB_MAP.get(_norm(order[raw_cls]))
                if canon is None or canon not in index:
                    continue
                tgt = index[canon]
                fixed = sanitize(tgt, coords, str(label_file), len(target))
                if fixed is not None:
                    labels.append((tgt, fixed))
                    stats['boxes'][(source, target[tgt])] += 1
            stats['images'][(source, src_split)] += 1
            records.append(Record(source, img_file, labels))
    # Conflict audit: opposite labels on heavily overlapping boxes.
    pairs = set()
    for a, b in OPPOSITES:
        if a in index and b in index:
            pairs |= {(index[a], index[b]), (index[b], index[a])}
    for rec in records:
        boxes = [((c, [(x[0] - x[2] / 2), (x[1] - x[3] / 2), (x[0] + x[2] / 2), (x[1] + x[3] / 2)]))
                 for c, x in rec.labels]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                (c1, b1), (c2, b2) = boxes[i], boxes[j]
                if (c1, c2) in pairs:
                    if iou_xyxy(b1, b2) > 0.5:
                        conflicts.append({'image': rec.image.name, 'source': source,
                                          'a': target[c1], 'b': target[c2]})
    return records


class BKTree:
    """Hamming-radius search without O(n^2): finds all stored hashes within
    max_bits of a query. Pure stdlib; 10k images build in ~a minute."""

    def __init__(self, max_bits: int):
        self.max_bits = max_bits
        self.root = None  # (value, {dist: child})

    @staticmethod
    def _dist(a: int, b: int) -> int:
        return bin(a ^ b).count('1')

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = [value, {}]
            return
        node = self.root
        while True:
            dist = self._dist(value, node[0])
            if dist == 0:
                return
            child = node[1].get(dist)
            if child is None:
                node[1][dist] = [value, {}]
                return
            node = child

    def query(self, value: int) -> list[int]:
        found = []
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            dist = self._dist(value, node[0])
            if dist <= self.max_bits:
                found.append(node[0])
            for edge, child in node[1].items():
                if dist - self.max_bits <= edge <= dist + self.max_bits:
                    stack.append(child)
        return found


def cluster_near_duplicates(records: list[Record], max_bits: int) -> list[list[int]]:
    parent = list(range(len(records)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_psha: dict[str, int] = {}
    for i, rec in enumerate(records):
        if rec.psha in by_psha:
            union(i, by_psha[rec.psha])
        else:
            by_psha[rec.psha] = i
    tree = BKTree(max_bits)
    seen_hash: dict[int, int] = {}
    for i, rec in enumerate(records):
        for hit in tree.query(rec.dh):
            union(i, seen_hash[hit])
        if rec.dh not in seen_hash:
            seen_hash[rec.dh] = i
            tree.add(rec.dh)
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return list(groups.values())


def build(args: argparse.Namespace) -> dict:
    import yaml

    import build_public_ppe as v1
    target = list(getattr(args, 'target_classes', None) or CLASSES)
    unknown = [c for c in target if c not in V1_CANON and c not in set(RB_MAP.values())]
    if not target or unknown:
        raise ValueError(f'Bad --target-classes {target}; known: sorted V1_CANON/RB_MAP names')
    viol_set = {target.index(n) for n in ('no_helmet', 'no_vest') if n in target}
    stats = {'images': collections.Counter(), 'boxes': collections.Counter()}
    conflicts: list[dict] = []
    records: list[Record] = []
    if args.rf_dir and args.coco_dir:
        rf_recs = v1.load_rf(Path(args.rf_dir))
        coco_recs = v1.load_coco(Path(args.coco_dir))
        for rec in rf_recs + coco_recs:
            labels = []
            for c, box in rec.labels:
                canon = v1.CLASSES[c]
                if canon not in target:
                    continue
                labels.append((target.index(canon), box))
            for c, _ in labels:
                stats['boxes'][(rec.source, target[c])] += 1
            stats['images'][(rec.source, rec.split)] += 1
            records.append(Record(rec.source, rec.image, labels))
    if args.rb_dir:
        records += load_yolo_dir(Path(args.rb_dir), 'roboflow_combined', stats, conflicts, target)
    if not records:
        raise SystemExit('No source dirs given. Pass --rb-dir and/or --rf-dir + --coco-dir.')
    for rec in records:
        rec.dh = dhash(rec.image)
        rec.psha = pixel_sha(rec.image)
    groups = cluster_near_duplicates(records, args.near_bits)
    exact_dups = sum(len(g) - 1 for g in groups
                     if len({records[i].psha for i in g}) == 1)
    # Violation-first sampling over whole clusters (deterministic).
    rng = random.Random(17)
    # Violation-first sampling over whole clusters (deterministic): clusters
    # with more violation boxes go first, so the budget fills with no_helmet
    # / no_vest content before clean frames.
    scored = []
    for g in groups:
        n_viol = sum(1 for i in g for c, _ in records[i].labels if c in viol_set)
        scored.append((0 if n_viol else 1, -n_viol, len(g), g))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    # Untouched holdout: whole scene clusters carved BEFORE sampling/split,
    # stratified by min-class key. Never used until the final evaluation.
    holdout_frac = float(getattr(args, 'holdout_frac', 0.0) or 0.0)
    holdout_ids: set[int] = set()
    holdout_groups: list[list[int]] = []
    if holdout_frac > 0:
        h_rng = random.Random(18)
        tmp_by_key: dict[int, list[list[int]]] = collections.defaultdict(list)
        for _, _, _, g in scored:
            key = min({c for i in g for c, _ in records[i].labels}, default=9)
            tmp_by_key[key].append(g)
        for key in sorted(tmp_by_key):
            clusters = tmp_by_key[key]
            h_rng.shuffle(clusters)
            take = round(len(clusters) * holdout_frac)
            for g in clusters[:take]:
                holdout_groups.append(g)
                holdout_ids.update(g)
    picked: list[int] = []
    seen_boxes: collections.Counter = collections.Counter()
    for _, _, _, g in scored:
        if g[0] in holdout_ids:
            continue
        if len(picked) + len(g) > args.max_images:
            continue
        picked += g
        for i in g:
            for c, _ in records[i].labels:
                seen_boxes[c] += 1
    # Stratified split by cluster: whole groups, class-balanced round-robin.
    by_class: dict[int, list[list[int]]] = collections.defaultdict(list)
    for _, _, _, g in scored:
        if g[0] not in picked:
            continue
        key = min({c for i in g for c, _ in records[i].labels}, default=9)
        by_class[key].append(g)
    splits: dict[str, list[int]] = {'train': [], 'val': [], 'test': []}
    frac = {'train': 0.8, 'val': 0.1, 'test': 0.1}
    for key in sorted(by_class):
        clusters = by_class[key]
        rng.shuffle(clusters)
        n = len(clusters)
        n_test = max(1, round(n * frac['test'])) if n >= 3 else 0
        n_val = max(1, round(n * frac['val'])) if n - n_test >= 2 else 0
        splits['test'] += [i for g in clusters[:n_test] for i in g]
        splits['val'] += [i for g in clusters[n_test:n_test + n_val] for i in g]
        splits['train'] += [i for g in clusters[n_test + n_val:] for i in g]
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output not empty: {out}')
    counts, per_class, manifest = {}, collections.Counter(), []
    for split, idxs in splits.items():
        for k, i in enumerate(sorted(idxs)):
            rec = records[i]
            img_d, lab_d = out / split / 'images', out / split / 'labels'
            img_d.mkdir(parents=True, exist_ok=True)
            lab_d.mkdir(parents=True, exist_ok=True)
            name = f'{rec.source}_{k:06d}{rec.image.suffix.lower()}'
            shutil.copyfile(rec.image, img_d / name)
            (lab_d / f'{Path(name).stem}.txt').write_text(''.join(
                f'{c} ' + ' '.join(f'{v:.6f}' for v in box) + '\n' for c, box in rec.labels))
            counts[split] = counts.get(split, 0) + 1
            for c, _ in rec.labels:
                per_class[target[c]] += 1
            manifest.append({'image': f'{split}/images/{name}', 'source': rec.source,
                             'split': split, 'objects': len(rec.labels),
                             'original': rec.image.name})
    holdout_counts: dict[str, int] = {}
    holdout_dir = getattr(args, 'holdout', None)
    if holdout_groups:
        holdout_dir = Path(holdout_dir) if holdout_dir else Path(str(out) + '_holdout')
        if holdout_dir.exists() and any(holdout_dir.iterdir()):
            raise FileExistsError(f'Holdout output not empty: {holdout_dir}')
        for k, i in enumerate(sorted(i for g in holdout_groups for i in g)):
            rec = records[i]
            img_d, lab_d = holdout_dir / 'images', holdout_dir / 'labels'
            img_d.mkdir(parents=True, exist_ok=True)
            lab_d.mkdir(parents=True, exist_ok=True)
            name = f'{rec.source}_{k:06d}{rec.image.suffix.lower()}'
            shutil.copyfile(rec.image, img_d / name)
            (lab_d / f'{Path(name).stem}.txt').write_text(''.join(
                f'{c} ' + ' '.join(f'{v:.6f}' for v in box) + '\n' for c, box in rec.labels))
            holdout_counts['holdout'] = holdout_counts.get('holdout', 0) + 1
        (holdout_dir / 'data.yaml').write_text(yaml.safe_dump(
            {'path': str(holdout_dir.resolve()), 'images': 'images',
             'nc': len(target), 'names': target},
            allow_unicode=True, sort_keys=False))
        (holdout_dir / 'HOLDOUT_README.md').write_text(
            '# Untouched holdout\n\nWhole scene clusters, carved before sampling. '
            'Use ONLY for the final evaluation — never train or tune on it.\n')
    if min(counts.get(s, 0) for s in ('train', 'val', 'test')) < 1:
        raise ValueError(f'Degenerate split: {counts}')
    for cls in range(len(target)):
        if seen_boxes[cls] < args.min_per_class:
            raise ValueError(f'Too few {target[cls]} boxes: {seen_boxes[cls]} < {args.min_per_class}')
    (out / 'data.yaml').write_text(yaml.safe_dump(
        {'path': str(out.resolve()), 'train': 'train/images', 'val': 'val/images',
         'test': 'test/images', 'nc': len(target), 'names': target},
        allow_unicode=True, sort_keys=False))
    report = {
        'counts': counts, 'per_class': dict(per_class),
        'images_per_source_split': {f'{s}/{sp}': n for (s, sp), n in stats['images'].items()},
        'boxes_per_source_class': {f'{s}/{c}': n for (s, c), n in stats['boxes'].items()},
        'scene_clusters': len(groups),
        'exact_duplicate_images_skipped': exact_dups,
        'near_bits': args.near_bits,
        'label_conflicts': conflicts[:200], 'label_conflict_count': len(conflicts),
        'classes': target, 'seed': 17,
        'holdout_frac': holdout_frac, 'holdout_counts': holdout_counts,
        'holdout_dir': str(holdout_dir) if holdout_groups else None,
        'license': 'CC BY 4.0 per source; see CREDITS.md',
        'warnings': ['Splits are scene-cluster independent, not video-id independent.',
                     'Near-duplicate threshold is perceptual, review borderline cases.',
                     'No smoking class; respirators deferred to a separate model.'],
    }
    (out / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out / 'CREDITS.md').write_text(
        '# V3 sources (CC BY 4.0)\n\n'
        '- Roboflow Universe Projects, Personal Protective Equipment - Combined Model\n'
        '  https://universe.roboflow.com/roboflow-universe-projects/personal-protective-equipment-combined-model\n'
        '- LibreYOLO construction-safety-gsnvb (RF100) + keremberke construction-safety-object-detection (v1 set)\n')
    print('V3 dataset:', str(out.resolve()))
    print(json.dumps({'counts': counts, 'per_class': dict(per_class),
                      'clusters': len(groups), 'conflicts': len(conflicts)}, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='datasets/ppe_v3')
    p.add_argument('--rb-dir', default=None, help='Roboflow YOLO export root (needs API key download first)')
    p.add_argument('--rf-dir', default=None)
    p.add_argument('--coco-dir', default=None)
    p.add_argument('--max-images', type=int, default=8000)
    p.add_argument('--min-per-class', type=int, default=200)
    p.add_argument('--near-bits', type=int, default=NEAR_BITS_DEFAULT)
    p.add_argument('--target-classes', nargs='+', default=None,
                   help='Ordered target classes (default: helmet no_helmet vest no_vest). '
                        'Non-default sets need decoder/threshold/test updates + base weights.')
    p.add_argument('--holdout-frac', type=float, default=0.0,
                   help='Fraction of scene clusters carved into an untouched holdout '
                        'before sampling (0 = off).')
    p.add_argument('--holdout', default=None,
                   help='Holdout output dir (default: <output>_holdout).')
    build(p.parse_args())


if __name__ == '__main__':
    main()
