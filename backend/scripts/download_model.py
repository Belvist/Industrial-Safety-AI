"""Download pinned ONNX artifacts without executing third-party model code."""

import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Versioned model registry: filename -> (download URL or None, SHA-256, provenance).
# Standard entries are fetched from their canonical URLs. Custom entries
# (locally trained) have url=None: the file must be placed manually, and its
# SHA is strictly verified — never silently trusted.
# Select with MODEL_PATH env (e.g. MODEL_PATH=models/ppe-v4cloud.onnx).
MODELS = {
    "ppe.onnx": (
        "https://huggingface.co/Hexmon/vyra-yolo-ppe-detection/resolve/08895b33d95d2587423ebe4f7c1b9c41beebd642/best.onnx",
        "b99fed37bae1d111ddb622a0ef9077d42918a4daebd4a0bffbd8faaba273f246",
        "Hexmon vyra-yolo-ppe-detection (third-party baseline, see THIRD_PARTY.md)",
    ),
    "person.onnx": (
        "https://huggingface.co/webml/yolov8n/resolve/85bc8d7ab30d4065a41909d756c42819b67b4388/onnx/yolov8n.onnx",
        "190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d",
        "Ultralytics YOLOv8n person (third-party, see THIRD_PARTY.md)",
    ),
    "ppe-v4.onnx": (
        None,
        "688bbf4d68217d1ff722aac06b2bae0a0296b6e5938d4b29f68623edd2d09f88",
        "local V4: 1608 imgs, 80 epochs MPS, seed 17 (training/runs/industrial-ppe-v4)",
    ),
    "ppe-v4cloud.onnx": (
        None,
        "f02ceb50ff1c2b4b41351e356c797dab866e5d900f4766b67b7cf542f6a55c46",
        "local V4-cloud: 5059 imgs, 80 epochs A10, seed 17 "
        "(training/runs/industrial-ppe-v4cloud)",
    ),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def default_env_file() -> Path:
    return ROOT / ".env"


def default_journal_file() -> Path:
    return ROOT / "models" / ".selection.json"


def read_journal(journal: Path) -> dict:
    import json

    if not journal.is_file():
        return {}
    try:
        data = json.loads(journal.read_text())
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def write_journal(journal: Path, record: dict) -> None:
    import json

    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps(record, indent=1))


def set_env_model(env_file: Path, value: str) -> None:
    """Set MODEL_PATH in a .env file (create if missing, replace if present)."""
    lines = env_file.read_text().splitlines() if env_file.is_file() else []
    out, seen = [], False
    for line in lines:
        if line.strip().startswith("MODEL_PATH="):
            out.append(f"MODEL_PATH={value}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"MODEL_PATH={value}")
    env_file.write_text("\n".join(out) + "\n")


def check_path(path: Path, download: bool = True) -> str:
    """Verify one registry file. Returns 'verified'. Raises otherwise."""
    entry = MODELS.get(path.name)
    if entry is None:
        raise SystemExit(
            f"Unknown model file {path.name}: not in the versioned registry. "
            f"Register its SHA in backend/scripts/download_model.py first.")
    url, checksum, provenance = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == checksum:
        print(f"Verified: {path} ({provenance})")
        return "verified"
    if path.exists():
        raise RuntimeError(
            f"{path} checksum mismatch (expected {checksum[:12]}… [{provenance}]). "
            f"Move it aside before installing, or rollback with --rollback."
        )
    if url is None or not download:
        raise SystemExit(
            f"{path} is a custom model with no download URL [{provenance}]. "
            f"Place the verified file manually, then re-run.")
    temporary = path.with_suffix(".download")
    print(f"Downloading {path.name}. Model licenses/provenance: THIRD_PARTY.md")
    try:
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as dest:
            while chunk := source.read(1024 * 1024):
                dest.write(chunk)
        if digest(temporary) != checksum:
            raise RuntimeError("Model checksum mismatch; refusing to install.")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    print("Model verified:", path)
    return "downloaded"


def do_select(target: Path, settings, env_file: Path, journal_file: Path) -> dict:
    """Switch the active model: verify target, back up current, journal, .env.

    Backup + journal record the PREVIOUS selection, so --rollback returns to
    the previous version — never to the same binary.
    """
    current = settings.resolve(settings.model_path)
    # Full normalization on BOTH sides: symlinked dirs (macOS /tmp) and
    # dot-segments must compare equal, otherwise re-select loops forever.
    # NOTE: relative targets resolve against the REPO root, never CWD.
    target = Path(target)
    target = (target if target.is_absolute() else ROOT / target).resolve()
    current = current.resolve() if current.exists() else current.absolute()
    entry = MODELS.get(target.name)
    if entry is None:
        raise SystemExit(f"Unknown model file {target.name}: not in the registry.")
    if target == current and target.is_file() and digest(target) == entry[1]:
        return {"selected": str(target), "changed": False,
                "note": "already selected and verified"}
    if not target.is_file() or digest(target) != entry[1]:
        raise SystemExit(
            f"Target {target} missing or checksum mismatch "
            f"(expected {entry[1][:12]}…). Refusing to select.")
    record = read_journal(journal_file)
    if current.is_file():
        backup = current.with_name(current.name + ".select-bak")
        backup.write_bytes(current.read_bytes())
        record["previous"] = {"path": str(current),
                              "sha": digest(current),
                              "backup": str(backup)}
    record["current"] = {"path": str(target), "sha": entry[1]}
    write_journal(journal_file, record)
    set_env_model(env_file, str(target) if target.is_absolute()
                  else os.path.relpath(target, ROOT))
    return {"selected": str(target), "changed": True,
            "previous": record.get("previous", {}).get("path")}


def do_rollback(settings, env_file: Path, journal_file: Path) -> dict:
    """Return to the previously selected model version (journal-driven)."""
    record = read_journal(journal_file)
    prev = record.get("previous")
    if not prev or not prev.get("backup"):
        raise SystemExit("No previous selection recorded: nothing to roll back to.")
    backup = Path(prev["backup"])
    if not backup.is_file() or digest(backup) != prev["sha"]:
        raise SystemExit(
            f"Rollback backup {backup} missing or altered. Refusing.")
    dest = Path(prev["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(backup.read_bytes())
    set_env_model(env_file, str(dest) if dest.is_absolute()
                  else os.path.relpath(dest, ROOT))
    record["current"] = {"path": str(dest), "sha": prev["sha"]}
    record.pop("previous", None)  # single-level undo; re-select to go forward
    write_journal(journal_file, record)
    print(f"Rolled back to: {dest} (sha {prev['sha'][:12]}…). Restart the server.")
    return {"restored": str(dest)}


def main(argv=None, settings=None):
    import argparse

    if settings is None:
        from app.core.config import settings as live_settings
        settings = live_settings

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollback", action="store_true",
        help="Return to the previously selected model version (journal-driven).",
    )
    parser.add_argument(
        "--select", default=None, metavar="MODEL_FILE",
        help="Verify + activate a registry model (backup, journal, .env).",
    )
    parsed = parser.parse_args(argv)
    if parsed.select:
        print(do_select(Path(parsed.select), settings,
                        default_env_file(), default_journal_file()))
        return
    if parsed.rollback:
        print(do_rollback(settings, default_env_file(), default_journal_file()))
        return
    model_path = settings.resolve(settings.model_path)
    person_needed = getattr(settings, "use_separate_person_model", True)
    check_path(model_path)
    if person_needed:
        check_path(settings.resolve(settings.person_model_path))
    else:
        print("Single-model mode: person.onnx not required, skipped.")


if __name__ == "__main__":
    main()
