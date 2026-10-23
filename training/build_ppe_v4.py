"""Build the V4 violation-focused PPE dataset from raw MP4 footage.

Full path: MP4 -> frame extraction -> draft autolabel -> mandatory QA ->
scene_id split -> licenses + manifest. Rejects the three failure modes found
in review of the V3 pipeline:

1. Raw video was never accepted (old builder wanted ready YOLO dirs).
   Here each video is one `video_id`: `footage_v4/<video_id>/` holds exactly
   one video file + LICENSE.json + optional hand-checked `labels/`.
2. Audit-file mismatch (`audit.json` vs `audit_leaks.json` gate in
   TRAIN_V3.py). This builder writes BOTH files, byte-identical.
3. pHash-only grouping let dissimilar frames of one video leak across
    splits. Here the split key is `scene_id` (hard guarantee): every video of
    one shoot shares a scene and can never split across train/val/holdout.
    pHash stays as a cross-split similarity audit (warning), exact pixel
    duplicates across splits are a hard failure.

V4 classes (fixed order): helmet, no_helmet, vest, no_vest, person.

Footage layout (footage_v4/ is git-ignored, private by default):

    footage_v4/clip_yard_morning/
        clip.mp4            # exactly one video file per dir
        LICENSE.json        # {"origin": "self-recorded"|"public",
                            #  "license": "owned"|"CC BY 4.0"|...,
                            #  "source_url": null|"...", "consent": true,
                            #  "scene_id": "shoot_yard_20261012" (optional,
                            #   default: video dir name). All clips of one
                            #   shoot MUST share one scene_id.}
        labels/f000120.txt  # hand-checked YOLO labels (optional but required
                            # for every frame that has no confirmed label)

Stages (default: all):

    python training/build_ppe_v4.py --footage footage_v4 --output datasets/ppe_v4 \\
        --holdout datasets/ppe_v4_holdout --min-per-class 150 --max-images 600

    --stage extract    MP4 -> staging frames only
    --stage autolabel  draft labels via ONNX weights (needs --autolabel-weights)
    --stage build      QA + split + outputs (needs confirmed labels)
    --self-test        synthetic-videos end-to-end check in tmp, no footage,
                       no weights, no network. Must stay green.

Draft policy: autolabel output is a DRAFT. Build fails on any draft frame
unless --allow-drafts is passed explicitly (and drafts are then flagged in
manifest.json). No silent training on unreviewed boxes.

Old-data guard: --forbid-dirs (default: all v1/v3 + v3-clean image dirs)
pixel-matches any V4 frame -> hard failure. Review/fn_* crops and the old
test/val splits must NEVER enter V4 training.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# Local geometry/hash helpers (stdlib + cv2 + numpy only) so this module
# imports in the backend CI env, which has no PIL/PyYAML. Semantics match
# training/build_ppe_v3.py; hashes are self-consistent within a build run
# (forbid set and V4 frames are hashed by the SAME functions here).
EDGE_TOL = 0.005


def sanitize(cls: int, coords: list[float], where: str, n_classes: int) -> list[float] | None:
    if not 0 <= cls < n_classes:
        raise ValueError(f'{where}: invalid target class {cls}')
    if len(coords) != 4 or any(not -EDGE_TOL <= v <= 1 + EDGE_TOL for v in coords):
        raise ValueError(f'{where}: bbox out of range {coords}')
    x, y, w, h = coords
    if w <= 0 or h <= 0:
        raise ValueError(f'{where}: non-positive size {coords}')
    x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
    x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
    if x1 >= x2 - 1e-6 or y1 >= y2 - 1e-6:
        return None
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * \
        max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter > 0 else 0.0


def dhash(path: Path) -> int:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'Undecodable image: {path}')
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_LINEAR)
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (1 if small[r, c] > small[r, c + 1] else 0)
    return bits


def pixel_sha(path: Path) -> str:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Undecodable image: {path}')
    return hashlib.sha256(img.tobytes() + str(img.shape).encode()).hexdigest()


def write_data_yaml(path: Path, root: Path, splits: dict[str, str]) -> None:
    """Minimal Ultralytics data.yaml without a PyYAML dependency."""
    lines = [f'path: {root.resolve()}']
    for key, val in splits.items():
        lines.append(f'{key}: {val}')
    lines.append(f'nc: {len(CLASSES)}')
    lines.append(f'names: [{", ".join(CLASSES)}]')
    path.write_text('\n'.join(lines) + '\n')


def find_conflicts(video_id: str, name: str,
                   labels: list[tuple[int, list[float]]]) -> list[dict]:
    """Opposite-label pairs (helmet/no_helmet, vest/no_vest, either order)
    with IoU > 0.5 on a single frame. Must resolve to zero before training."""
    boxes = [(c, [x[0] - x[2] / 2, x[1] - x[3] / 2,
                  x[0] + x[2] / 2, x[1] + x[3] / 2]) for c, x in labels]
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (c1, b1), (c2, b2) = boxes[i], boxes[j]
            if (CLASSES[c1], CLASSES[c2]) in OPPOSITES and iou_xyxy(b1, b2) > 0.5:
                out.append({'video': video_id, 'image': name,
                            'a': CLASSES[c1], 'b': CLASSES[c2]})
    return out


def audit_cross_split(fingerprints: dict[str, list[tuple[int, str]]],
                      split_of: dict[str, str],
                      near_bits: int) -> tuple[int, list[dict]]:
    """All-pairs cross-split duplicate audit.

    fingerprints: video_id -> [(dhash, pixel_sha)] per frame.
    - pixel SHA equality across splits = IDENTICAL frames -> hard fail basis.
      (dHash equality alone proves nothing: distinct images can collide, so a
      dHash-only fail would block legit data.)
    - dHash within near_bits (but different pixels) = visually similar ->
      warning, once per video pair.
    Positional zip() would be WRONG here: videos differ in frame counts and
    offsets, so the same frame at different positions would never meet.
    Limit: pixel SHA catches exact copies under one decoding, not re-encoded
    near-duplicates — those stay covered by the dHash warning.
    """
    exact, near = 0, []
    vids = sorted(fingerprints)
    for i, a in enumerate(vids):
        for b in vids[i + 1:]:
            if split_of.get(a) == split_of.get(b):
                continue
            pair_near = False
            for ha, pa in fingerprints[a]:
                for hb, pb in fingerprints[b]:
                    if pa == pb:
                        exact += 1
                    elif not pair_near and bin(ha ^ hb).count('1') <= near_bits:
                        near.append({'a': a, 'b': b})
                        pair_near = True
    return exact, near

CLASSES = ['helmet', 'no_helmet', 'vest', 'no_vest', 'person']
OPPOSITES = {('no_helmet', 'helmet'), ('helmet', 'no_helmet'),
             ('no_vest', 'vest'), ('vest', 'no_vest')}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv'}
IMG_Q = 90

DEFAULT_FORBID = [
    'ppe_public_v3/train', 'ppe_public_v3/val', 'ppe_public_v3/test',
    'datasets/ppe_v3_clean/train', 'datasets/ppe_v3_clean/val',
    'datasets/ppe_v3_clean/test',
]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Stage 0: ingest + extract
# --------------------------------------------------------------------------

def read_license(video_dir: Path) -> dict:
    lic_path = video_dir / 'LICENSE.json'
    if not lic_path.is_file():
        raise SystemExit(f'{video_dir}: missing LICENSE.json '
                         f'(origin/license/source_url/consent required).')
    try:
        lic = json.loads(lic_path.read_text())
    except ValueError as exc:
        raise SystemExit(f'{lic_path}: invalid JSON: {exc}') from exc
    if lic.get('origin') not in {'self-recorded', 'public'}:
        raise SystemExit(f'{lic_path}: origin must be self-recorded|public.')
    if lic.get('consent') is not True:
        raise SystemExit(f'{lic_path}: consent must be true (people agreed).')
    if lic.get('origin') == 'public':
        if not lic.get('source_url') or not str(lic.get('license', '')).upper().startswith('CC'):
            raise SystemExit(f'{lic_path}: public footage needs source_url + '
                             f'CC BY 4.0 (or more permissive) license.')
    scene = lic.get('scene_id', video_dir.name)
    if not isinstance(scene, str) or not scene.strip() or '/' in scene or '\\' in scene:
        raise SystemExit(f'{lic_path}: scene_id must be a non-empty name without slashes.')
    lic['scene_id'] = scene.strip()
    return lic


def find_video(video_dir: Path) -> Path:
    vids = sorted(p for p in video_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if len(vids) != 1:
        raise SystemExit(f'{video_dir}: need exactly one video '
                         f'{sorted(VIDEO_EXTS)}, found {len(vids)}.')
    return vids[0]


def extract_video(video: Path, out_images: Path, extract_fps: float) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f'Cannot open video: {video}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(src_fps / extract_fps))
    out_images.mkdir(parents=True, exist_ok=True)
    frames, idx, saved = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            name = f'f{idx:06d}.jpg'
            cv2.imwrite(str(out_images / name), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, IMG_Q])
            frames.append({'image': name, 'frame_index': idx,
                           't_sec': round(idx / src_fps, 3)})
            saved += 1
        idx += 1
    cap.release()
    if not frames:
        raise SystemExit(f'{video}: no frames extracted.')
    print(f'  {video.parent.name}: {saved} frames (every {step}, src {src_fps:.1f} fps)',
          flush=True)
    return frames


def stage_extract(args: argparse.Namespace) -> dict:
    footage = Path(args.footage)
    staging = Path(args.staging)
    if not footage.is_dir():
        raise SystemExit(f'No footage root: {footage} (see V4_DATASET_PLAN.md).')
    videos = sorted(p for p in footage.iterdir()
                    if p.is_dir() and not p.name.startswith('_'))
    if not videos:
        raise SystemExit(f'{footage}: no <video_id>/ dirs.')
    plan = {'videos': {}}
    for vdir in videos:
        lic = read_license(vdir)
        video = find_video(vdir)
        out_images = staging / vdir.name / 'images'
        frames = extract_video(video, out_images, args.extract_fps)
        meta = {'video_id': vdir.name, 'scene_id': lic['scene_id'],
                'source_video': video.name,
                'video_sha256': sha_file(video), 'license': lic,
                'frames': frames}
        (staging / vdir.name / 'frames.jsonl').write_text(
            '\n'.join(json.dumps({**{'video_id': vdir.name}, **f}) for f in frames))
        plan['videos'][vdir.name] = meta
    (staging / 'extract_plan.json').write_text(json.dumps(plan, indent=2))
    print(f'Extracted {len(videos)} video(s) -> {staging}', flush=True)
    return plan


# --------------------------------------------------------------------------
# Stage 1: draft autolabel (optional, needs ONNX weights)
# --------------------------------------------------------------------------

def stage_autolabel(args: argparse.Namespace) -> None:
    if not args.autolabel_weights:
        print('No --autolabel-weights: skipping draft autolabel. '
              'Provide hand-checked labels/ per video dir.', flush=True)
        return
    sys.path.insert(0, str(HERE.parent / 'backend'))
    from app.inference.local_onnx import LocalOnnxProvider

    staging = Path(args.staging)
    provider = LocalOnnxProvider(
        args.autolabel_weights, 0.4, 2,
        person_model_path=args.person_model,
        thresholds={c: 0.0 for c in CLASSES}, floor=0.25, roi_mode='off')
    provider.load()
    index = {name: i for i, name in enumerate(CLASSES)}
    per_video = [d for d in staging.iterdir()
                 if d.is_dir() and (d / 'images').is_dir()]
    if not per_video:
        raise SystemExit(f'{staging}: nothing extracted (run --stage extract first).')
    for vdir in sorted(per_video):
        hand = (Path(args.footage) / vdir.name / 'labels'
                if not args.self_test_tmp else None)
        draft_dir = vdir / 'labels_draft'
        draft_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for img in sorted((vdir / 'images').glob('*.jpg')):
            if hand is not None and (hand / f'{img.stem}.txt').is_file():
                continue  # hand-checked label wins, no draft needed
            frame = cv2.imread(str(img))
            lines = []
            for det in provider.predict_sync(frame):
                if det.class_name not in index:
                    continue
                b = det.bbox
                cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
                lines.append(f'{index[det.class_name]} {cx:.6f} {cy:.6f} '
                             f'{b.x2 - b.x1:.6f} {b.y2 - b.y1:.6f}  # conf {det.confidence:.2f}')
            (draft_dir / f'{img.stem}.txt').write_text('\n'.join(lines) + '\n' if lines else '')
            n += 1
        print(f'  {vdir.name}: {n} draft label(s) in labels_draft/ (REVIEW REQUIRED)',
              flush=True)


# --------------------------------------------------------------------------
# Stage 2: QA + video_id split + outputs
# --------------------------------------------------------------------------

def load_confirmed_labels(args: argparse.Namespace, video_id: str,
                          images: list[str]) -> tuple[dict[str, list], dict[str, bool]]:
    """Hand labels win; drafts fill gaps and are flagged. Every frame needs one."""
    hand_dir = Path(args.footage) / video_id / 'labels'
    draft_dir = Path(args.staging) / video_id / 'labels_draft'
    labels, is_draft = {}, {}
    for name in images:
        stem = Path(name).stem
        hand = hand_dir / f'{stem}.txt'
        draft = draft_dir / f'{stem}.txt'
        if hand.is_file():
            labels[name] = hand.read_text().splitlines()
            is_draft[name] = False
        elif draft.is_file():
            labels[name] = draft.read_text().splitlines()
            is_draft[name] = True
        else:
            raise SystemExit(f'{video_id}/{name}: no confirmed label and no draft. '
                             f'Add footage_v4/{video_id}/labels/{stem}.txt '
                             f'or run --stage autolabel first.')
    return labels, is_draft


def parse_labels(video_id: str, name: str, lines: list[str]) -> list[tuple[int, list[float]]]:
    out = []
    for lineno, line in enumerate(lines, 1):
        body = line.split('#', 1)[0].strip()
        if not body:
            continue
        parts = body.split()
        if len(parts) != 5:
            raise SystemExit(f'{video_id}/{name}:{lineno}: expected 5 columns.')
        try:
            cls, coords = int(parts[0]), [float(v) for v in parts[1:]]
        except ValueError:
            raise SystemExit(f'{video_id}/{name}:{lineno}: non-numeric values.')
        if not 0 <= cls < len(CLASSES):
            raise SystemExit(f'{video_id}/{name}:{lineno}: class {cls} out of 0..{len(CLASSES) - 1}.')
        fixed = sanitize(cls, coords, f'{video_id}/{name}:{lineno}', len(CLASSES))
        if fixed is not None:
            out.append((cls, fixed))
    return out


def draw_boxes(frame, labels: list[tuple[int, list[float]]]):
    colors = [(80, 200, 80), (60, 60, 240), (240, 180, 40), (240, 60, 200), (200, 200, 200)]
    h, w = frame.shape[:2]
    for cls, (x, y, bw, bh) in labels:
        x1, y1, x2, y2 = int((x - bw / 2) * w), int((y - bh / 2) * h), \
            int((x + bw / 2) * w), int((y + bh / 2) * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls % len(colors)], 2)
        cv2.putText(frame, CLASSES[cls], (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[cls % len(colors)], 1)
    return frame


def contact_sheet(samples: list[tuple[str, object, list]], path: Path, cols: int = 4):
    import numpy as np

    if not samples:
        return
    thumbs = []
    for name, frame, labels in samples[:16]:
        vis = draw_boxes(frame.copy(), labels)
        cv2.putText(vis, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        thumbs.append(cv2.resize(vis, (320, 240)))
    rows = [thumbs[i:i + cols] for i in range(0, len(thumbs), cols)]
    width = cols * 320
    sheet = np.zeros((len(rows) * 240, width, 3), dtype=np.uint8)
    for r, row in enumerate(rows):
        for c, t in enumerate(row):
            sheet[r * 240:(r + 1) * 240, c * 320:(c + 1) * 320] = t
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def forbid_pixel_shas(forbid_dirs: list[str]) -> set[str]:
    """Pixel SHAs of old data that must never enter V4 (exact-match guard).

    A non-empty --forbid-dirs that hashes ZERO images is a vacuous guard
    (missing dirs) and fails loudly. Pass --forbid-dirs with no entries to
    opt out explicitly (self-test does this: no old data exists there).
    """
    shas: set[str] = set()
    n = 0
    for entry in forbid_dirs:
        d = Path(entry)
        if not d.is_absolute():
            d = HERE.parent / d
        img_d = d / 'images' if (d / 'images').is_dir() else d
        if not img_d.is_dir():
            print(f'  forbid dir missing: {entry}', flush=True)
            continue
        for img in sorted(img_d.iterdir()):
            if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            try:
                shas.add(pixel_sha(img))
                n += 1
            except Exception:
                continue
    print(f'  forbid guard: {n} old image(s) hashed from {len(forbid_dirs)} dir(s)', flush=True)
    if forbid_dirs and n == 0:
        raise SystemExit('Forbid guard vacuous: --forbid-dirs given but 0 old images '
                         'hashed (all dirs missing?). Point it at real old-data dirs, '
                         'or pass --forbid-dirs with no entries to opt out explicitly.')
    return shas


def stage_build(args: argparse.Namespace) -> dict:
    staging = Path(args.staging)
    plan_path = staging / 'extract_plan.json'
    if not plan_path.is_file():
        raise SystemExit(f'{staging}: no extract_plan.json (run --stage extract first).')
    plan = json.loads(plan_path.read_text())
    video_ids = sorted(plan['videos'])
    scenes: dict[str, list[str]] = collections.defaultdict(list)
    for vid in video_ids:
        scenes[plan['videos'][vid]['scene_id']].append(vid)
    scene_ids = sorted(scenes)
    need = 3 if args.holdout_frac > 0 else 2
    if len(scene_ids) < need:
        raise SystemExit(f'Need >= {need} scenes for '
                         f'train/val{"+holdout" if args.holdout_frac > 0 else ""} '
                         f'splits, found {len(scene_ids)}: {scene_ids}. '
                         f'All clips of one shoot must share scene_id; shoot '
                         f'separate sessions (see V4_DATASET_PLAN.md).')

    # Load + QA every frame.
    records: dict[str, list] = {}       # video_id -> [(name, labels, draft)]
    conflicts: list[dict] = []
    for vid in video_ids:
        img_dir = staging / vid / 'images'
        names = sorted(p.name for p in img_dir.glob('*.jpg'))
        labels_raw, drafts = load_confirmed_labels(args, vid, names)
        if any(drafts.values()) and not args.allow_drafts:
            n = sum(1 for v in drafts.values() if v)
            raise SystemExit(f'{vid}: {n} unreviewed DRAFT label(s). Review them into '
                             f'footage_v4/{vid}/labels/ or pass --allow-drafts explicitly.')
        recs = []
        for name in names:
            labels = parse_labels(vid, name, labels_raw[name])
            conflicts.extend(find_conflicts(vid, name, labels))
            recs.append((name, labels, drafts[name]))
        records[vid] = recs
    if conflicts:
        raise SystemExit(f'{len(conflicts)} opposite-label conflicts (IoU>0.5), '
                         f'first: {conflicts[0]}. Resolve to zero before training.')

    # Old-data guard (exact pixel matches).
    forbid = forbid_pixel_shas(args.forbid_dirs)
    if forbid:
        for vid, recs in records.items():
            for name, _, _ in recs:
                if pixel_sha(staging / vid / 'images' / name) in forbid:
                    raise SystemExit(f'{vid}/{name}: pixel-identical to old-dataset '
                                     f'data (forbid guard). Old test/val/review crops '
                                     f'must NEVER train V4.')

    # Split by scene_id (hard guarantee: one shoot owns exactly one split),
    # deterministic. Videos inherit their scene's split.
    rng = random.Random(args.seed)
    order = scene_ids[:]
    rng.shuffle(order)
    n_hold = max(1, round(len(order) * args.holdout_frac)) if args.holdout_frac > 0 else 0
    holdout_scenes = order[:n_hold]
    rest = order[n_hold:]
    n_val = max(1, round(len(rest) * args.val_frac))
    if len(rest) - n_val < 1:
        raise SystemExit(f'val split would empty train ({len(rest)} scenes). '
                         f'Add scenes or lower --val-frac.')
    val_scenes, train_scenes = rest[:n_val], rest[n_val:]
    scene_split = {s: 'train' for s in train_scenes}
    scene_split.update({s: 'val' for s in val_scenes})
    scene_split.update({s: 'holdout' for s in holdout_scenes})
    splits = {'train': [v for s in train_scenes for v in scenes[s]],
              'val': [v for s in val_scenes for v in scenes[s]],
              'holdout': [v for s in holdout_scenes for v in scenes[s]]}
    train_vids, val_vids, holdout_vids = splits['train'], splits['val'], splits['holdout']
    assert len(set(train_vids) | set(val_vids) | set(holdout_vids)) == len(video_ids)
    assert not (set(train_vids) & set(val_vids) or set(train_vids) & set(holdout_vids)
                or set(val_vids) & set(holdout_vids))

    # Budget + min-per-class gates (train+val only; holdout untouched).
    per_class: collections.Counter = collections.Counter()
    n_frames = 0
    for vid in train_vids + val_vids:
        for _, labels, _ in records[vid]:
            n_frames += 1
            for c, _ in labels:
                per_class[CLASSES[c]] += 1
    if n_frames > args.max_images:
        raise SystemExit(f'{n_frames} frames > --max-images {args.max_images}. '
                         f'Lower --extract-fps or raise the budget deliberately.')
    for cls in ('no_helmet', 'no_vest'):
        if per_class.get(cls, 0) < args.min_per_class:
            raise SystemExit(f'Too few {cls}: {per_class.get(cls, 0)} < '
                             f'{args.min_per_class} (by design: shoot more violations).')

    # Cross-split audit: pixel-identical frames = fail, dHash-similar = warn.
    split_of = {v: s for s, vs in splits.items() for v in vs}
    fingerprints = {v: [(dhash(staging / v / 'images' / n),
                         pixel_sha(staging / v / 'images' / n))
                        for n, _, _ in records[v]]
                    for v in video_ids}
    exact_across, near_across = audit_cross_split(fingerprints, split_of, args.near_bits)
    if exact_across:
        raise SystemExit(f'{exact_across} pixel-identical frame(s) across splits: '
                         f'scene_id isolation broken, investigate.')

    # Write outputs.
    out, holdout = Path(args.output), Path(args.holdout)
    for d in (out, holdout):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(f'Output not empty: {d} (move it first).')
    manifest, sheet_samples = [], []
    counts: dict[str, int] = {}
    for split in ('train', 'val'):
        for vid in splits[split]:
            meta = plan['videos'][vid]
            for name, labels, draft in records[vid]:
                img_d, lab_d = out / split / 'images', out / split / 'labels'
                img_d.mkdir(parents=True, exist_ok=True)
                lab_d.mkdir(parents=True, exist_ok=True)
                new = f'{vid}_{Path(name).stem}.jpg'
                shutil.copyfile(staging / vid / 'images' / name, img_d / new)
                (lab_d / f'{Path(new).stem}.txt').write_text(''.join(
                    f'{c} ' + ' '.join(f'{v:.6f}' for v in box) + '\n'
                    for c, box in labels))
                counts[split] = counts.get(split, 0) + 1
                manifest.append({'image': f'{split}/images/{new}', 'split': split,
                                 'video_id': vid, 'scene_id': meta['scene_id'],
                                 'source_video': meta['source_video'],
                                 'video_sha256': meta['video_sha256'],
                                 'license': meta['license'], 'draft_label': draft,
                                 'objects': len(labels)})
                if len(sheet_samples) < 16 and labels:
                    sheet_samples.append((new, cv2.imread(str(img_d / new)), labels))
    write_data_yaml(out / 'data.yaml', out, {'train': 'train/images', 'val': 'val/images'})
    hold_counts: dict[str, int] = {}
    for vid in splits['holdout']:
        meta = plan['videos'][vid]
        for name, labels, draft in records[vid]:
            img_d, lab_d = holdout / 'images', holdout / 'labels'
            img_d.mkdir(parents=True, exist_ok=True)
            lab_d.mkdir(parents=True, exist_ok=True)
            new = f'{vid}_{Path(name).stem}.jpg'
            shutil.copyfile(staging / vid / 'images' / name, img_d / new)
            (lab_d / f'{Path(new).stem}.txt').write_text(''.join(
                f'{c} ' + ' '.join(f'{v:.6f}' for v in box) + '\n'
                for c, box in labels))
            hold_counts['holdout'] = hold_counts.get('holdout', 0) + 1
            manifest.append({'image': f'images/{new}', 'split': 'holdout',
                             'video_id': vid, 'scene_id': meta['scene_id'],
                             'source_video': meta['source_video'],
                             'video_sha256': meta['video_sha256'],
                             'license': meta['license'], 'draft_label': draft,
                             'objects': len(labels)})
    write_data_yaml(holdout / 'data.yaml', holdout, {'val': 'images', 'test': 'images'})
    (holdout / 'HOLDOUT_README.md').write_text(
        '# Untouched holdout\n\nWhole scenes, carved before sampling. '
        'Use ONLY for the final evaluation — never train or tune on it.\n')
    contact_sheet(sheet_samples, out / 'contact_sheet.jpg')
    credits = ['# V4 sources\n']
    for vid in video_ids:
        lic = plan['videos'][vid]['license']
        line = f'- {vid}: origin={lic["origin"]} license={lic.get("license")}'
        if lic.get('source_url'):
            line += f' source={lic["source_url"]}'
        credits.append(line + '\n')
    (out / 'CREDITS.md').write_text(''.join(credits))
    audit = {
        'classes': CLASSES, 'seed': args.seed, 'split_key': 'scene_id',
        'scenes': {s: scene_split[s] for s in scene_ids},
        'videos': {v: split_of[v] for v in video_ids},
        'counts': counts, 'per_class': dict(per_class),
        'holdout_counts': hold_counts,
        'exact_duplicates_across_splits': exact_across,
        'duplicate_check': 'pixel-sha equality hard-fails, dhash similarity warns',
        'near_similar_video_pairs_across_splits': near_across[:50],
        'near_similar_pair_count': len(near_across),
        'near_bits': args.near_bits, 'label_conflict_count': 0,
        'draft_labels': sum(1 for m in manifest if m['draft_label']),
        'license': 'per-video, see CREDITS.md',
        'warnings': ['Splits are scene_id independent by construction: all clips '
                     'of one shoot share one split.',
                     'Review near-similar cross-split pairs before publishing.',
                     'No smoking class; respirators deferred to a separate model.'],
    }
    blob = json.dumps(audit, indent=2, ensure_ascii=False)
    (out / 'audit.json').write_text(blob)
    (out / 'audit_leaks.json').write_text(blob)  # same gate file TRAIN_V3.py checks
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('V4 dataset:', str(out.resolve()), flush=True)
    print(json.dumps({'videos': splits, 'counts': counts,
                      'per_class': dict(per_class),
                      'holdout': hold_counts,
                      'near_pairs': len(near_across)}, indent=2), flush=True)
    return audit


# --------------------------------------------------------------------------
# Self-test: synthetic videos, no footage / weights / network
# --------------------------------------------------------------------------

def make_synthetic_video(path: Path, seed: int, n_frames: int = 10,
                         size: tuple[int, int] = (160, 120)) -> None:
    """Deterministic toy clip. Layout varies by seed (distinct dHash per
    video, like distinct real scenes); a moving dot varies frames in time."""
    import numpy as np

    rng = random.Random(seed)
    base = rng.randint(30, 220)
    rx, ry = rng.randint(5, 60), rng.randint(5, 40)
    rw, rh = rng.randint(30, 70), rng.randint(40, 80)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, size)
    if not writer.isOpened():
        raise SystemExit(f'Self-test: VideoWriter failed for {path} (codec missing).')
    for k in range(n_frames):
        frame = np.full((size[1], size[0], 3), base, np.uint8)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (200, 200, 200), -1)
        cx = 10 + (k * 25 + seed) % (size[0] - 20)
        cv2.circle(frame, (cx, 60), 8, (50, 50, 50), -1)
        writer.write(frame)
    writer.release()


def run_self_test() -> dict:
    """End-to-end pipeline on 4 synthetic labeled videos / 3 scenes in tmp.

    Two clips share one shoot (scene_id) — they must land in one split.
    Returns audit. No footage, weights, or network.
    """
    tmp = Path(tempfile.mkdtemp(prefix='v4_selftest_'))
    footage, staging = tmp / 'footage', tmp / 'staging'
    fixture = (('clip_day_close', 11, 'shoot_day'), ('clip_day_wide', 12, 'shoot_day'),
               ('clip_evening', 22, 'shoot_evening'), ('clip_yard', 33, 'shoot_yard'))
    for vid, seed, scene in fixture:
        vdir = footage / vid
        vdir.mkdir(parents=True)
        make_synthetic_video(vdir / 'clip.mp4', seed, n_frames=12)
        (vdir / 'LICENSE.json').write_text(json.dumps(
            {'origin': 'self-recorded', 'license': 'owned',
             'source_url': None, 'consent': True, 'scene_id': scene}))
    args = argparse.Namespace(footage=str(footage), staging=str(staging),
                              extract_fps=2.0, autolabel_weights=None,
                              person_model=None, self_test_tmp=str(tmp))
    plan = stage_extract(args)
    # Hand labels for every extracted frame: alternate violation / clean.
    for vid, meta in plan['videos'].items():
        lab_d = footage / vid / 'labels'
        lab_d.mkdir(exist_ok=True)
        for k, f in enumerate(meta['frames']):
            if k % 3 == 0:
                body = '1 0.375 0.416667 0.375 0.666667\n4 0.375 0.5 0.5 0.833333\n'
            elif k % 3 == 1:
                body = '0 0.375 0.208333 0.25 0.25\n2 0.375 0.583333 0.375 0.5\n4 0.375 0.5 0.5 0.833333\n'
            else:
                body = '3 0.375 0.583333 0.375 0.5\n4 0.375 0.5 0.5 0.833333\n'
            (lab_d / f'{Path(f["image"]).stem}.txt').write_text(body)
    args_build = argparse.Namespace(
        footage=str(footage), staging=str(staging), seed=17,
        holdout_frac=0.34, val_frac=0.5, max_images=600, min_per_class=2,
        near_bits=8, allow_drafts=False, forbid_dirs=[],
        output=str(tmp / 'ppe_v4'), holdout=str(tmp / 'ppe_v4_holdout'),
        self_test_tmp=str(tmp))
    audit = stage_build(args_build)
    # Assertions: scene isolation (shared shoot -> one split) + file parity.
    assert set(audit['scenes']) == {'shoot_day', 'shoot_evening', 'shoot_yard'}
    assert len(set(audit['scenes'].values())) == 3, audit['scenes']
    by_scene: dict[str, set] = collections.defaultdict(set)
    for vid, split in audit['videos'].items():
        scene = next(meta['scene_id'] for v, meta in plan['videos'].items() if v == vid)
        by_scene[scene].add(split)
    assert all(len(s) == 1 for s in by_scene.values()), dict(by_scene)
    assert (tmp / 'ppe_v4' / 'audit_leaks.json').read_text() == \
        (tmp / 'ppe_v4' / 'audit.json').read_text()
    assert (tmp / 'ppe_v4' / 'data.yaml').is_file()
    assert (tmp / 'ppe_v4_holdout' / 'HOLDOUT_README.md').is_file()
    manifest = json.loads((tmp / 'ppe_v4' / 'manifest.json').read_text())
    assert manifest and all(m['video_id'] and m['scene_id'] for m in manifest)
    print(f'SELF-TEST PASS ({tmp})', flush=True)
    return audit


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--footage', default='footage_v4')
    p.add_argument('--staging', default=None,
                   help='Frame staging dir (default: <footage>/_staging).')
    p.add_argument('--stage', default='all', choices=['all', 'extract', 'autolabel', 'build'])
    p.add_argument('--extract-fps', type=float, default=0.5,
                   help='Frames per second to keep (plan default: 1 frame / 2 s).')
    p.add_argument('--autolabel-weights', default=None,
                   help='ONNX weights for draft labels (optional).')
    p.add_argument('--person-model', default='models/person.onnx')
    p.add_argument('--allow-drafts', action='store_true',
                   help='Accept unreviewed draft labels (flagged in manifest).')
    p.add_argument('--output', default='datasets/ppe_v4')
    p.add_argument('--holdout', default='datasets/ppe_v4_holdout')
    p.add_argument('--holdout-frac', type=float, default=0.15)
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--max-images', type=int, default=600)
    p.add_argument('--min-per-class', type=int, default=150)
    p.add_argument('--near-bits', type=int, default=8)
    p.add_argument('--forbid-dirs', nargs='*', default=DEFAULT_FORBID)
    p.add_argument('--self-test', action='store_true',
                   help='Synthetic end-to-end check, ignores footage/output args.')
    args = p.parse_args()
    if args.self_test:
        run_self_test()
        return
    args.self_test_tmp = None
    if args.staging is None:
        args.staging = str(Path(args.footage) / '_staging')
    if args.stage in ('all', 'extract'):
        stage_extract(args)
    if args.stage in ('all', 'autolabel'):
        stage_autolabel(args)
    if args.stage in ('all', 'build'):
        stage_build(args)


if __name__ == '__main__':
    main()
