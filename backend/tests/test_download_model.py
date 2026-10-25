"""download_model.py: registry, refusals, rollback — no network touches.

Happy-path downloads are NOT covered here (CI must not fetch 100MB+).
`settings` is stubbed per-test (the module binds the global singleton at
import, so env vars would not apply). Covered: registry shape, digest(),
unknown-name refusal, missing custom file refusal, rollback refusal without
backup, rollback success in tmp.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location(
    "download_model", ROOT / "backend" / "scripts" / "download_model.py")
dm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dm)


class _S:
    def __init__(self, model, person="models/person.onnx", single=False):
        self.model_path, self.person_model_path = model, person
        self.use_separate_person_model = not single

    @staticmethod
    def resolve(v):
        p = Path(v)
        return p if p.is_absolute() else ROOT / p


def test_registry_entries_have_sha_and_provenance():
    for name, entry in dm.MODELS.items():
        url, checksum, provenance = entry
        assert name.endswith(".onnx")
        assert len(checksum) == 64 and all(
            c in "0123456789abcdef" for c in checksum)
        assert provenance.strip()
    assert "ppe.onnx" in dm.MODELS and "person.onnx" in dm.MODELS


def test_local_known_files_match_registry():
    """Integration: real binaries vs registry. Skipped in clean checkouts
    (weights are git-ignored) — never fails CI for missing files."""
    present = [p for p in (ROOT / "models").glob("*.onnx") if p.is_file()]
    if not present:
        pytest.skip("no .onnx weights locally (git-ignored)")
    for path in present:
        entry = dm.MODELS.get(path.name)
        if entry is None:
            continue  # unrelated local file, not our registry's business
        assert dm.digest(path) == entry[1], f"{path.name} drifted from registry"


def _fixture_registry(tmp_path, monkeypatch):
    """Temp registry + files: proves verification logic without real weights."""
    fix = tmp_path / "fix.onnx"
    fixp = tmp_path / "fixp.onnx"
    fix.write_bytes(b"fixture-model" * 100)
    fixp.write_bytes(b"fixture-person" * 100)
    registry = {
        "fix.onnx": (None, dm.digest(fix), "test fixture"),
        "fixp.onnx": (None, dm.digest(fixp), "test fixture"),
    }
    monkeypatch.setattr(dm, "MODELS", registry)
    return fix, fixp


def test_registry_verifies_fixture_files(tmp_path, monkeypatch):
    fix, fixp = _fixture_registry(tmp_path, monkeypatch)
    dm.main([], _S(str(fix), str(fixp)))  # must exit 0, no network


def test_registry_rejects_tampered_fixture(tmp_path, monkeypatch):
    fix, fixp = _fixture_registry(tmp_path, monkeypatch)
    with fix.open("r+b") as f:
        f.seek(10)
        f.write(b"XX")
    with pytest.raises((RuntimeError, SystemExit)):
        dm.main([], _S(str(fix), str(fixp)))


def test_digest_matches_hashlib(tmp_path):
    blob = tmp_path / "b.bin"
    blob.write_bytes(b"abc123" * 1000)
    assert dm.digest(blob) == hashlib.sha256(blob.read_bytes()).hexdigest()


def test_unknown_model_name_refused():
    with pytest.raises(SystemExit):
        dm.main([], _S("models/custom-mystery.onnx"))


def test_missing_custom_model_errors_without_network(tmp_path):
    with pytest.raises(SystemExit):
        dm.main([], _S(str(tmp_path / "ppe-v4cloud.onnx")))


def test_single_mode_skips_person_model(tmp_path, monkeypatch):
    fix, _fixp = _fixture_registry(tmp_path, monkeypatch)
    # person file missing entirely — must not matter in single mode
    dm.main([], _S(str(fix), str(tmp_path / "absent-person.onnx"), single=True))


def test_rollback_without_journal_refused(tmp_path):
    with pytest.raises(SystemExit):
        dm.do_rollback(_S("models/ppe-v4.onnx"), tmp_path / ".env",
                       tmp_path / "journal.json")


def test_select_is_idempotent_same_model(tmp_path, monkeypatch):
    src = ROOT / "models" / "ppe-v4.onnx"
    if not src.is_file():
        pytest.skip("local PPE weights absent (not in git)")
    real = tmp_path / "real"
    real.mkdir()
    target = real / "ppe-v4.onnx"
    target.write_bytes(src.read_bytes())
    monkeypatch.setattr(dm, "MODELS", {
        "ppe-v4.onnx": (None, dm.digest(target), "test"),
    })
    env, journal = tmp_path / ".env", tmp_path / "journal.json"
    first = dm.do_select(target, _S(str(tmp_path / "other.onnx")), env, journal)
    assert first["changed"] is True
    # same file via a spelled-out path with dot segments: still a no-op
    alias = tmp_path / "." / "real" / ".." / "real" / target.name
    again = dm.do_select(alias, _S(str(target)), env, journal)
    assert again["changed"] is False


def test_select_rollback_roundtrip_previous_version(tmp_path, monkeypatch):
    src_a = ROOT / "models" / "ppe-v4.onnx"
    src_b = ROOT / "models" / "ppe-v4cloud.onnx"
    if not (src_a.is_file() and src_b.is_file()):
        pytest.skip("local PPE weights absent (not in git)")
    a = tmp_path / "ppe-v4.onnx"
    b = tmp_path / "ppe-v4cloud.onnx"
    a.write_bytes(src_a.read_bytes())
    b.write_bytes(src_b.read_bytes())
    monkeypatch.setattr(dm, "MODELS", {
        "ppe-v4.onnx": (None, dm.digest(a), "test A"),
        "ppe-v4cloud.onnx": (None, dm.digest(b), "test B"),
    })
    env, journal = tmp_path / ".env", tmp_path / "journal.json"
    # select B while A active -> journal records A as previous
    dm.do_select(b, _S(str(a)), env, journal)
    assert "ppe-v4cloud.onnx" in env.read_text()
    # tamper active file, rollback must restore PREVIOUS version (A), not B
    b.write_bytes(b"tampered")
    dm.do_rollback(_S(str(b)), env, journal)
    assert dm.digest(a) == dm.MODELS["ppe-v4.onnx"][1]
    assert "ppe-v4.onnx" in env.read_text()


def test_select_rejects_tampered_target(tmp_path, monkeypatch):
    src = ROOT / "models" / "ppe-v4.onnx"
    if not src.is_file():
        pytest.skip("local PPE weights absent (not in git)")
    bad = tmp_path / "ppe-v4.onnx"
    bad.write_bytes(b"tampered")
    monkeypatch.setattr(dm, "MODELS", {
        "ppe-v4.onnx": (None, dm.digest(src), "test"),
    })
    with pytest.raises(SystemExit):
        dm.do_select(bad, _S(str(tmp_path / "ppe.onnx")), tmp_path / ".env",
                     tmp_path / "journal.json")
