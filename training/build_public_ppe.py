"""Build Industrial Safety AI 7-class YOLO dataset from two CC-BY-4.0 sources.

Dependencies: pip install huggingface_hub pillow pyyaml
Command: python training/build_public_ppe.py --output datasets/ppe_public_v1

This script converts existing human annotations; it never hallucinates boxes,
replaces negative PPE labels, or labels all people as noncompliant.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

CLASSES = ["person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask"]
RF_SOURCE = ["helmet", "no-helmet", "no-vest", "person", "vest"]
RF_REMAP = {0: 1, 1: 2, 2: 4, 3: 0, 4: 3}
COCO_TO_TARGET = {
    "person": "person", "hardhat": "helmet", "helmet": "helmet",
    "no-hardhat": "no_helmet", "no-helmet": "no_helmet",
    "safety vest": "vest", "vest": "vest", "no-safety vest": "no_vest",
    "no-vest": "no_vest", "mask": "mask", "no-mask": "no_mask",
    "no mask": "no_mask",
}
RF_REPO = "LibreYOLO/construction-safety-gsnvb"
COCO_REPO = "keremberke/construction-safety-object-detection"
HF = "https://huggingface.co/datasets/"
SPLITS = {"train": "train", "valid": "val", "test": "test"}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_label(name: str) -> str:
    return " ".join(name.lower().replace("_", "-").split())


# Tolerance for pixel->normalized rounding at frame edges (e.g. x+w/2 = 1.0009).
# Overflows within it are clipped; anything bigger is a broken mapping and stays loud.
EDGE_TOL = 0.005
SANITIZE_STATS = {"clipped_boxes": 0, "dropped_boxes": 0}


def _validate_ann(class_id: int, coords: list[float], where: str) -> None:
    if class_id < 0 or class_id >= len(CLASSES):
        raise ValueError(f"{where}: invalid target class {class_id}")
    if len(coords) != 4 or any(not 0 <= x <= 1 for x in coords):
        raise ValueError(f"{where}: bbox out of 0..1 {coords}")
    x, y, w, h = coords
    if w <= 0 or h <= 0 or x - w / 2 < -1e-5 or y - h / 2 < -1e-5 or x + w / 2 > 1 + 1e-5 or y + h / 2 > 1 + 1e-5:
        raise ValueError(f"{where}: invalid bbox geometry {coords}")


def _sanitize_ann(class_id: int, coords: list[float], where: str) -> list[float] | None:
    """Clip rounding-level edge overflow; drop degenerate boxes; keep loud errors loud."""
    if class_id < 0 or class_id >= len(CLASSES):
        raise ValueError(f"{where}: invalid target class {class_id}")
    if len(coords) != 4 or any(not -EDGE_TOL <= x <= 1 + EDGE_TOL for x in coords):
        raise ValueError(f"{where}: bbox out of range {coords}")
    x, y, w, h = coords
    if w <= 0 or h <= 0:
        raise ValueError(f"{where}: non-positive bbox size {coords}")
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    if min(x1, y1) < -EDGE_TOL or max(x2, y2) > 1 + EDGE_TOL:
        raise ValueError(f"{where}: invalid bbox geometry {coords}")
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        SANITIZE_STATS["clipped_boxes"] += 1
    nw, nh = cx2 - cx1, cy2 - cy1
    if nw <= 1e-6 or nh <= 1e-6:
        SANITIZE_STATS["dropped_boxes"] += 1
        return None
    return [(cx1 + cx2) / 2, (cy1 + cy2) / 2, nw, nh]


def _extract_zip_safely(file: Path, to: Path) -> None:
    """No absolute paths, .. traversal, symlinks or ZIP bomb expansion."""
    with zipfile.ZipFile(file) as z:
        infos = z.infolist()
        if sum(i.file_size for i in infos) > 500_000_000:
            raise ValueError(f"Archive too big: {file}")
        for item in infos:
            member = Path(item.filename)
            if member.is_absolute() or ".." in member.parts or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe ZIP entry: {item.filename}")
        z.extractall(to)


@dataclass
class ImageRecord:
    split: str
    source: str
    image: Path
    labels: list[tuple[int, list[float]]]


def load_rf(root: Path) -> list[ImageRecord]:
    import yaml

    config = yaml.safe_load((root / "data.yaml").read_text())
    assert config["names"] == RF_SOURCE, f"Unexpected RF100 class order: {config['names']}"
    records: list[ImageRecord] = []
    for src_split, tgt_split in SPLITS.items():
        imgs = root / src_split / "images"
        labs = root / src_split / "labels"
        if not imgs.is_dir() or not labs.is_dir():
            raise FileNotFoundError(f"RF100 missing split: {src_split}")
        for file in sorted(imgs.iterdir()):
            if file.suffix.lower() not in EXTS:
                continue
            label_file = labs / f"{file.stem}.txt"
            # Standard YOLO convention: absent .txt means an image without any
            # labeled objects (a background negative), NOT missing PPE on person.
            labels = []
            for line in (label_file.read_text().splitlines() if label_file.is_file() else []):
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"Invalid YOLO label in {label_file}")
                old_class = int(parts[0])
                if old_class not in RF_REMAP:
                    raise ValueError(f"Unknown RF100 class {old_class}")
                cls, coords = RF_REMAP[old_class], [float(v) for v in parts[1:]]
                fixed = _sanitize_ann(cls, coords, str(label_file))
                if fixed is not None:
                    labels.append((cls, fixed))
            records.append(ImageRecord(tgt_split, "rf100", file, labels))
    return records


def load_coco(root: Path) -> list[ImageRecord]:
    records = []
    for src_split, tgt_split in SPLITS.items():
        annotation_files = list((root / src_split).rglob("_annotations.coco.json"))
        if len(annotation_files) != 1:
            raise ValueError(f"COCO {src_split} expected 1 annotation file, found {len(annotation_files)}")
        meta = json.loads(annotation_files[0].read_text())
        catmap = {}
        for category in meta["categories"]:
            normalized = _source_label(category["name"])
            if normalized in COCO_TO_TARGET:
                catmap[int(category["id"])] = CLASSES.index(COCO_TO_TARGET[normalized])
        required = {"person", "helmet", "no_helmet", "vest", "no_vest", "mask", "no_mask"}
        covered = {CLASSES[i] for i in catmap.values()}
        if covered != required:
            raise ValueError(f"COCO {src_split} unexpected classes: missing {required - covered}")
        images = {int(img["id"]): img for img in meta["images"]}
        anns = collections.defaultdict(list)
        for ann in meta["annotations"]:
            cid = int(ann["category_id"])
            if cid not in catmap or ann.get("iscrowd", 0):
                continue
            iid = int(ann["image_id"])
            if iid not in images:
                raise ValueError(f"COCO orphan annotation: image {iid}")
            img = images[iid]
            iw, ih = int(img["width"]), int(img["height"])
            x, y, w, h = map(float, ann["bbox"])
            coords = [(x + w / 2) / iw, (y + h / 2) / ih, w / iw, h / ih]
            fixed = _sanitize_ann(catmap[cid], coords, f"COCO image {iid} annotation {ann['id']}")
            if fixed is not None:
                anns[iid].append((catmap[cid], fixed))
        folder = annotation_files[0].parent
        for iid, img in images.items():
            name = Path(img["file_name"]).name
            file = folder / name
            if not file.is_file():
                hits = list((root / src_split).rglob(name))
                if len(hits) != 1:
                    raise FileNotFoundError(f"COCO missing/ambiguous image {src_split}: {name}")
                file = hits[0]
            records.append(ImageRecord(tgt_split, "coco", file, anns[iid]))
    return records


def _download(url: str, to: Path) -> Path:
    if to.exists() and to.stat().st_size > 1000:
        return to
    to.parent.mkdir(parents=True, exist_ok=True)
    tmp = to.with_suffix(to.suffix + ".part")
    print(f"Downloading {url.rsplit('/', 1)[-1]} ...", flush=True)
    with urllib.request.urlopen(url, timeout=120) as src, tmp.open("wb") as out:
        shutil.copyfileobj(src, out)
    tmp.replace(to)
    return to


def download_sources(cache: Path) -> tuple[Path, Path]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install: python -m pip install huggingface_hub pillow pyyaml") from exc
    rf = cache / "rf100"
    snapshot_download(
        repo_id=RF_REPO,
        repo_type="dataset",
        local_dir=rf,
        allow_patterns=["data.yaml", "train/images/*", "train/labels/*",
                        "valid/images/*", "valid/labels/*", "test/images/*", "test/labels/*"],
        max_workers=6,
    )
    coco = cache / "coco"
    for split in SPLITS:
        zip_file = cache / "coco_archives" / f"{split}.zip"
        _download(f"{HF}{COCO_REPO}/resolve/main/data/{split}.zip", zip_file)
        split_dir = coco / split
        if not list(split_dir.rglob("_annotations.coco.json")):
            _extract_zip_safely(zip_file, split_dir)
    return rf, coco


def assemble(records: list[ImageRecord], out: Path, min_per_split: int = 1) -> dict:
    """Preserve upstream splits; forbid byte-identical and pixel-identical leakage."""
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output not empty; move it first: {out}")
    # Respect upstream split while keeping test/val on duplicate collisions.
    priority = {"test": 0, "val": 1, "train": 2}
    records = sorted(records, key=lambda x: (priority[x.split], x.source, str(x.image)))
    out.mkdir(parents=True, exist_ok=True)
    filehash_seen = {}
    counts = collections.Counter()
    class_counts = collections.Counter()
    duplicates = []
    manifest = []
    for idx, rec in enumerate(records):
        with Image.open(rec.image) as img:
            img.verify()
        with Image.open(rec.image) as img:
            rgb = img.convert("RGB")
            pixelhash = hashlib.sha256(rgb.tobytes() + str(rgb.size).encode()).hexdigest()
        previous = filehash_seen.get(pixelhash)
        if previous:
            duplicates.append({"image": str(rec.image), "previous": previous})
            continue
        filehash_seen[pixelhash] = {"split": rec.split, "image": str(rec.image)}
        image_to = out / rec.split / "images"
        label_to = out / rec.split / "labels"
        image_to.mkdir(parents=True, exist_ok=True)
        label_to.mkdir(parents=True, exist_ok=True)
        ext = rec.image.suffix.lower()
        if ext not in EXTS:
            raise ValueError(f"Unsupported image extension: {rec.image}")
        name = f"{rec.source}_{idx:06d}"
        shutil.copyfile(rec.image, image_to / f"{name}{ext}")
        target_txt = label_to / f"{name}.txt"
        target_txt.write_text("".join(
            str(cls) + " " + " ".join(f"{v:.7f}" for v in coords) + "\n"
            for cls, coords in rec.labels
        ))
        counts[rec.split] += 1
        for cls, _ in rec.labels:
            class_counts[CLASSES[cls]] += 1
        manifest.append({"image": f"{rec.split}/images/{name}{ext}",
                         "source": rec.source, "split": rec.split,
                         "objects": len(rec.labels), "original": rec.image.name,
                         "pixel_sha256": pixelhash})
    if min(counts[s] for s in ("train", "val", "test")) < min_per_split:
        raise ValueError(f"Empty/small split: {dict(counts)}")
    import yaml
    config = {"path": str(out.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
              "nc": len(CLASSES), "names": CLASSES}
    (out / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    report = {"counts": dict(counts), "per_class": dict(class_counts),
              "labels_total": sum(class_counts.values()), "removed_exact_duplicate_images": len(duplicates),
              "clipped_edge_boxes": SANITIZE_STATS["clipped_boxes"],
              "dropped_degenerate_boxes": SANITIZE_STATS["dropped_boxes"],
              "dropped_duplicates": duplicates[:100], "classes": CLASSES,
              "sources": {"rf100": RF_REPO, "coco": COCO_REPO},
              "license": "Both CC BY 4.0; attribute Roboflow 100 and Roboflow Universe Projects",
              "warnings": ["Upstream image splits are retained; near-duplicate video frames are NOT guaranteed independent.",
                           "Original source boxes have not been manually re-annotated or visually audited.",
                           "No smoking annotations in either source; this model cannot detect smoking.",
                           "Keep independent real-camera footage as final held-out evaluation."]}
    (out / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (out / "CREDITS.md").write_text(
        "# Source attribution (CC BY 4.0)\n\n"
        "- Roboflow 100, Construction Safety Gsnvb, https://huggingface.co/datasets/LibreYOLO/construction-safety-gsnvb\n"
        "- Roboflow Universe Projects, Construction Site Safety Dataset, https://huggingface.co/datasets/keremberke/construction-safety-object-detection\n"
        "- Original object boxes mapped to canonical names; no synthetic negative PPE boxes.\n"
        "- RF100 benchmark citation: Ciaglia et al., arXiv:2211.13523.\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/ppe_public_v1"))
    parser.add_argument("--cache", type=Path, default=Path("datasets/.source_cache"))
    parser.add_argument("--rf-dir", type=Path, default=None, help="Local YOLO source for offline tests")
    parser.add_argument("--coco-dir", type=Path, default=None, help="Local COCO extracted source for offline tests")
    args = parser.parse_args()
    if bool(args.rf_dir) != bool(args.coco_dir):
        parser.error("Pass BOTH --rf-dir and --coco-dir for offline preparation")
    SANITIZE_STATS["clipped_boxes"] = 0
    SANITIZE_STATS["dropped_boxes"] = 0
    rf, coco = (args.rf_dir, args.coco_dir) if args.rf_dir else download_sources(args.cache)
    records = load_rf(rf) + load_coco(coco)
    report = assemble(records, args.output)
    print("Built labeled dataset:", args.output.resolve())
    print(json.dumps({k: report[k] for k in ("counts", "per_class", "removed_exact_duplicate_images")}, indent=2))


if __name__ == "__main__":
    main()
