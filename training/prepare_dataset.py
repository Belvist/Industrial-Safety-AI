"""Build an Ultralytics dataset.yaml with clip-level train/val/test split.

Frames of one clip (one recording) never leak across splits.

Usage:
    python training/prepare_dataset.py ../datasets --out training/ppe-custom.yaml
"""

import argparse
import random
from pathlib import Path

CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask"]

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def validate_clip(clip: Path) -> tuple[int, dict[str, int]]:
    """Validate images + YOLO labels. Returns (labeled images, class counts)."""
    try:
        import cv2  # optional: pixel validation skipped without it
    except ImportError:
        cv2 = None
    counts: dict[str, int] = {}
    labeled = 0
    for image in sorted((clip / "images").iterdir()):
        if image.suffix.lower() not in IMG_EXTS:
            continue
        if image.stat().st_size == 0:
            raise SystemExit(f"Empty image file: {image}")
        if cv2 is not None and cv2.imread(str(image)) is None:
            raise SystemExit(f"Undecodable image: {image}")
        label = clip / "labels" / (image.stem + ".txt")
        if not label.is_file():
            continue
        labeled += 1
        for lineno, line in enumerate(label.read_text().splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                raise SystemExit(f"{label}:{lineno}: expected 5 columns, got {len(parts)}")
            try:
                cls = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                raise SystemExit(f"{label}:{lineno}: non-numeric values")
            if not 0 <= cls < len(CLASSES):
                raise SystemExit(f"{label}:{lineno}: class id {cls} out of range 0..{len(CLASSES) - 1}")
            if any(not 0.0 <= v <= 1.0 for v in coords):
                raise SystemExit(f"{label}:{lineno}: coordinates outside 0..1")
            counts[CLASSES[cls]] = counts.get(CLASSES[cls], 0) + 1
    return labeled, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Dataset root with <clip>/{images,labels}/")
    parser.add_argument("--out", default="training/ppe-custom.yaml")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    root = Path(args.root)
    clips = sorted(p for p in root.iterdir() if (p / "images").is_dir())
    if not clips:
        raise SystemExit(f"No <clip>/images/ folders in {root}. See datasets/README.md.")
    counts = {}
    class_totals: dict[str, int] = {}
    for clip in clips:
        names = {p.stem for p in (clip / "images").glob("*") if p.suffix.lower() in IMG_EXTS}
        labeled, class_counts = validate_clip(clip)
        for cls, num in class_counts.items():
            class_totals[cls] = class_totals.get(cls, 0) + num
        counts[clip.name] = (len(names), labeled)
    usable = [c for c in clips if counts[c.name][1] > 0]
    skipped = [c.name for c in clips if counts[c.name][1] == 0]
    if skipped:
        print(f"Skipped clips without labels: {skipped}")
    rng = random.Random(args.seed)
    rng.shuffle(usable)
    n = len(usable)
    if n < 3:
        raise SystemExit(
            f"Need at least 3 independent labeled clips, found {n}. "
            "Shoot separate sessions (see datasets/README.md); a single clip "
            "cannot serve as train/val/test."
        )
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    if n_train == 0 or n_val == 0 or n - n_train - n_val == 0:
        raise SystemExit(
            f"Empty split with {n} clips (train={n_train}, val={n_val}, "
            f"test={n - n_train - n_val}). Add more independent clips or adjust ratios."
        )
    split = (
        [("train", usable[:n_train]), ("val", usable[n_train:n_train + n_val]),
         ("test", usable[n_train + n_val:])]
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"path: {root.resolve()}", "", f"nc: {len(CLASSES)}", f"names: {CLASSES}", ""]
    for name, group in split:
        lines.append(f"{name}:")
        for clip in group:
            lines.append(f"  - {clip.name}/images")
        print(f"{name}: {len(group)} clips, "
              f"{sum(counts[c.name][0] for c in group)} images")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")
    print("Class distribution: " + ", ".join(
        f"{cls}={class_totals.get(cls, 0)}" for cls in CLASSES))
    # Validate the produced YAML with framework-independent checks.
    _validate_yaml(out, root)


def _validate_yaml(path: Path, root: Path) -> None:
    """Framework-independent YAML check: sections exist, are non-empty,
    directories resolve, nc/names match CLASSES."""
    text = path.read_text()
    assert f"nc: {len(CLASSES)}" in text, "nc mismatch"
    assert str(CLASSES) in text, "names mismatch"
    section = None
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("path:"):
            continue
        if line.strip().endswith(":") and not line.startswith((" ", "-")):
            section = line.strip()[:-1]
            found[section] = []
        elif line.strip().startswith("- ") and section in {"train", "val", "test"}:
            found[section].append(line.strip()[2:])
    for name in ("train", "val", "test"):
        if not found.get(name):
            raise SystemExit(f"YAML validation failed: empty {name} split in {path}")
        for entry in found[name]:
            if not (root / entry).is_dir():
                raise SystemExit(f"YAML validation failed: missing dir {root / entry}")
    print("YAML validated: train/val/test non-empty, all dirs resolve.")


if __name__ == "__main__":
    main()
