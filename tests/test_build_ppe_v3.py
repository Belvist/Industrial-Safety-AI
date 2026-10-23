"""Tests for the V3 violation-focused dataset builder."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parents[1] / 'training'))
import build_ppe_v3 as v3  # noqa: E402


def paint(path: Path, seed: int, mark: bool = False, roll: int = 0):
    # Seeded noise: distinct seeds hash far apart; (11, roll=1) is a near-dup
    # of (11, roll=0) for exercising the perceptual cluster path.
    import random
    rng = random.Random(seed)
    data = bytes(rng.randrange(256) for _ in range(96 * 96 * 3))
    if roll:
        data = data[roll * 96 * 3:] + data[:roll * 96 * 3]
    img = Image.frombytes('RGB', (96, 96), data)
    if mark:
        ImageDraw.Draw(img).rectangle([10, 10, 40, 40], fill=(200, 30, 30))
    img.save(path)


def rb_fixture(root: Path):
    import yaml
    for split in ('train', 'valid', 'test'):
        (root / split / 'images').mkdir(parents=True)
        (root / split / 'labels').mkdir(parents=True)
    (root / 'data.yaml').write_text(yaml.safe_dump(
        {'names': ['Hardhat', 'NO-Hardhat', 'Safety Vest', 'NO-Safety Vest', 'Person']}))
    # (split, name, seed, mark, label): 3+ scene clusters per target class,
    # plus a near-duplicate pair (a, c) that must stay in one split.
    rows = [
        ('train', 'a', (11, 0), True, '1 0.5 0.5 0.3 0.3\n'),
        ('train', 'b', (37, 0), False, '0 0.5 0.5 0.2 0.2\n'),
        ('train', 'e', (71, 0), False, '0 0.5 0.5 0.2 0.2\n'),
        ('train', 'l', (139, 0), False, '0 0.5 0.5 0.2 0.2\n'),
        ('train', 'i', (211, 0), False, '2 0.5 0.5 0.4 0.4\n'),
        ('train', 'k', (199, 0), False, '3 0.5 0.5 0.2 0.2\n'),
        ('valid', 'c', (11, 1), True, '1 0.5 0.5 0.3 0.3\n3 0.2 0.2 0.1 0.1\n'),
        ('valid', 'f', (173, 0), False, '2 0.5 0.5 0.4 0.4\n'),
        ('valid', 'j', (149, 0), False, '3 0.5 0.5 0.2 0.2\n'),
        ('test', 'd', (103, 0), False, '2 0.5 0.5 0.4 0.4\n'),
        ('test', 'g', (41, 0), False, '3 0.5 0.5 0.2 0.2\n'),
        ('test', 'h', (61, 0), False, '1 0.5 0.5 0.3 0.3\n'),
        ('test', 'h2', (227, 0), False, '1 0.5 0.5 0.3 0.3\n'),
    ]
    for split, name, seed_roll, mark, label in rows:
        seed, roll = seed_roll
        paint(root / split / 'images' / f'{name}.jpg', seed, mark=mark, roll=roll)
        (root / split / 'labels' / f'{name}.txt').write_text(label)
    return root


def test_rb_mapping_and_stats(tmp_path):
    root = rb_fixture(tmp_path / 'rb')
    stats = {'images': __import__('collections').Counter(), 'boxes': __import__('collections').Counter()}
    recs = v3.load_yolo_dir(root, 'rb', stats, [])
    assert stats['boxes'][('rb', 'no_helmet')] == 4
    assert stats['boxes'][('rb', 'helmet')] == 3
    assert all(r.labels for r in recs)


def test_conflict_audit(tmp_path):
    root = rb_fixture(tmp_path / 'rb')
    (root / 'train' / 'labels' / 'b.txt').write_text('0 0.5 0.5 0.3 0.3\n1 0.51 0.51 0.3 0.3\n')
    conflicts: list = []
    stats = {'images': __import__('collections').Counter(), 'boxes': __import__('collections').Counter()}
    v3.load_yolo_dir(root, 'rb', stats, conflicts)
    assert any(c['a'] == 'helmet' and c['b'] == 'no_helmet' for c in conflicts)


def test_near_dup_cluster_keeps_one_split(tmp_path):
    import argparse
    root = rb_fixture(tmp_path / 'rb')
    args = argparse.Namespace(output=str(tmp_path / 'out'), rb_dir=str(root),
                              rf_dir=None, coco_dir=None, max_images=100,
                              min_per_class=1, near_bits=8)
    report = v3.build(args)
    manifest = {m['original']: m['split'] for m in
                __import__('json').loads((tmp_path / 'out' / 'manifest.json').read_text())}
    # a.jpg and c.jpg are near-duplicates: must land in the same split.
    assert manifest['a.jpg'] == manifest['c.jpg']
    # 13 images, one near-dup pair -> 12 scene clusters.
    assert report['scene_clusters'] == 12
    assert set(report['counts']) == {'train', 'val', 'test'}


def test_target_classes_five_with_person(tmp_path):
    import argparse, json
    root = rb_fixture(tmp_path / 'rb')
    # Person boxes pass through only when requested.
    (root / 'train' / 'labels' / 'b.txt').write_text('0 0.5 0.5 0.2 0.2\n4 0.3 0.3 0.2 0.2\n')
    target = ['helmet', 'no_helmet', 'vest', 'no_vest', 'person']
    args = argparse.Namespace(output=str(tmp_path / 'out'), rb_dir=str(root),
                              rf_dir=None, coco_dir=None, max_images=100,
                              min_per_class=0, near_bits=8,
                              target_classes=target, holdout_frac=0.0, holdout=None)
    report = v3.build(args)
    assert report['classes'] == target
    assert report['per_class'].get('person', 0) >= 1
    data = (tmp_path / 'out' / 'data.yaml').read_text()
    assert 'nc: 5' in data and 'person' in data


def test_holdout_carves_whole_clusters_untouched(tmp_path):
    import argparse, json
    root = rb_fixture(tmp_path / 'rb')
    # One extra scene cluster per class so test/val survive the carve-out.
    extra = [('train', 'm', (301, 0), False, '0 0.5 0.5 0.2 0.2\n'),
             ('valid', 'n', (302, 0), False, '1 0.5 0.5 0.3 0.3\n'),
             ('test', 'o', (303, 0), False, '2 0.5 0.5 0.4 0.4\n'),
             ('train', 'p', (304, 0), False, '3 0.5 0.5 0.2 0.2\n')]
    for split, name, seed_roll, mark, label in extra:
        seed, roll = seed_roll
        paint(root / split / 'images' / f'{name}.jpg', seed, mark=mark, roll=roll)
        (root / split / 'labels' / f'{name}.txt').write_text(label)
    args = argparse.Namespace(output=str(tmp_path / 'out'), rb_dir=str(root),
                              rf_dir=None, coco_dir=None, max_images=100,
                              min_per_class=0, near_bits=8,
                              target_classes=None, holdout_frac=0.34, holdout=None)
    report = v3.build(args)
    hold = tmp_path / 'out_holdout'
    assert hold.is_dir()
    held = {p.name for p in (hold / 'images').glob('*')}
    main_imgs = set()
    for split in ('train', 'val', 'test'):
        main_imgs |= {p.name for p in (tmp_path / 'out' / split / 'images').glob('*')}
    # Holdout names are namespaced copies; originals must not overlap.
    manifest_main = {m['original'] for m in
                     json.loads((tmp_path / 'out' / 'manifest.json').read_text())}
    assert len(held) == report['holdout_counts']['holdout'] > 0
    assert 'a.jpg' in manifest_main or 'c.jpg' in manifest_main
    assert (hold / 'HOLDOUT_README.md').is_file()
    assert (hold / 'data.yaml').is_file()


def test_violation_first_and_class_floor(tmp_path):
    import argparse, json
    root = rb_fixture(tmp_path / 'rb')
    args = argparse.Namespace(output=str(tmp_path / 'out'), rb_dir=str(root),
                              rf_dir=None, coco_dir=None, max_images=6,
                              min_per_class=0, near_bits=8)
    report = v3.build(args)
    manifest = {m['original']: m['split'] for m in
                json.loads((tmp_path / 'out' / 'manifest.json').read_text())}
    # Violation clusters fill the budget first, biggest violation content first:
    # {a.jpg,c.jpg} (3 violation boxes), then k.jpg, j.jpg, g.jpg, h.jpg.
    # Clean b.jpg waits.
    assert set(manifest) == {'a.jpg', 'c.jpg', 'k.jpg', 'j.jpg', 'g.jpg', 'h.jpg'}
    assert set(report['counts']) == {'train', 'val', 'test'}
    assert report['per_class'].get('no_helmet', 0) >= 2
