"""V4 dataset pipeline: scene isolation, audit parity, old-data guard.

Runs on synthetic videos in tmp (no footage, no weights, no network) so the
full MP4 -> dataset path is verified in CI. Uses only stdlib + cv2 + numpy.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

import build_ppe_v4 as v4  # noqa: E402  (training/ added to path above)


def _load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_v3_gate", ROOT / "ppe_public_v3" / "TRAIN_V3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_self_test_passes():
    audit = v4.run_self_test()
    assert set(audit["videos"]) == {"clip_day_close", "clip_day_wide",
                                    "clip_evening", "clip_yard"}
    # Three shoots own three different splits; the two day clips (one shoot)
    # never separate.
    assert set(audit["scenes"]) == {"shoot_day", "shoot_evening", "shoot_yard"}
    assert len(set(audit["scenes"].values())) == 3
    assert (audit["videos"]["clip_day_close"]
            == audit["videos"]["clip_day_wide"])


def test_gate_accepts_both_audit_files(tmp_path):
    import json as _json

    resolve_gate = _load_train_module().resolve_gate
    both = tmp_path / "both"
    both.mkdir()
    (both / "audit.json").write_text(_json.dumps(_strict_audit()))
    (both / "audit_leaks.json").write_text(_json.dumps(_strict_audit()))
    assert resolve_gate(both).name == "audit_leaks.json"
    only_json = tmp_path / "only_json"
    only_json.mkdir()
    (only_json / "audit.json").write_text(_json.dumps(_strict_audit()))
    assert resolve_gate(only_json).name == "audit.json"
    with pytest.raises(SystemExit):
        resolve_gate(only_json / "missing")


def test_gate_rejects_empty_and_garbage_audit(tmp_path):
    resolve_gate = _load_train_module().resolve_gate
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "audit_leaks.json").write_text("{}")
    with pytest.raises(SystemExit):
        resolve_gate(empty)
    garbage = tmp_path / "garbage"
    garbage.mkdir()
    (garbage / "audit_leaks.json").write_text("not json {{{")
    with pytest.raises(SystemExit):
        resolve_gate(garbage)
    lst = tmp_path / "lst"
    lst.mkdir()
    (lst / "audit_leaks.json").write_text("[1, 2]")
    with pytest.raises(SystemExit):
        resolve_gate(lst)


def _strict_audit(**over):
    audit = {"counts": {"train": 10}, "label_conflict_count": 0,
             "exact_duplicates_across_splits": 0, "draft_labels": 0}
    audit.update(over)
    return audit


def test_gate_rejects_unsafe_audits(tmp_path):
    import json as _json

    mod = _load_train_module()
    good = tmp_path / "good"
    good.mkdir()
    (good / "audit_leaks.json").write_text(_json.dumps(_strict_audit()))
    assert mod.resolve_gate(good).name == "audit_leaks.json"
    for bad in ({"label_conflict_count": 2},
                {"exact_duplicates_across_splits": 1},
                {"draft_labels": 3}):
        d = tmp_path / f"bad_{sorted(bad)[0]}"
        d.mkdir()
        (d / "audit_leaks.json").write_text(_json.dumps(_strict_audit(**bad)))
        with pytest.raises(SystemExit):
            mod.resolve_gate(d)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "audit_leaks.json").write_text('{"counts": {"train": 10}}')
    with pytest.raises(SystemExit):
        mod.resolve_gate(incomplete)
    # Legacy shape (counts, no safety keys) stays grandfathered.
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "audit.json").write_text('{"counts": {"train": 1}, "errors": []}')
    assert mod._check_audit_content(legacy / "audit.json", strict=False).is_file()


def test_find_conflicts_both_orders():
    box = [0.5, 0.5, 0.4, 0.4]
    far = [0.1, 0.1, 0.05, 0.05]
    # helmet=0, no_helmet=1, vest=2, no_vest=3 (V4 order).
    assert len(v4.find_conflicts("v", "f.jpg", [(0, box), (1, box)])) == 1
    assert len(v4.find_conflicts("v", "f.jpg", [(1, box), (0, box)])) == 1
    assert len(v4.find_conflicts("v", "f.jpg", [(3, box), (2, box)])) == 1
    # Negatives: no overlap, same class, unrelated classes.
    assert v4.find_conflicts("v", "f.jpg", [(0, box), (1, far)]) == []
    assert v4.find_conflicts("v", "f.jpg", [(0, box), (0, box)]) == []
    assert v4.find_conflicts("v", "f.jpg", [(0, box), (2, box)]) == []


def test_cross_split_audit_compares_all_pairs():
    # Pixel-identical frames at swapped positions: zip() would miss them,
    # all-pairs must catch both (pixel SHA is the hard-fail basis).
    a, b = 0b00000000, 0b11111111
    exact, near = v4.audit_cross_split(
        {"a": [(a, "P"), (b, "Q")], "b": [(b, "Q"), (a, "P")]},
        {"a": "train", "b": "holdout"}, 1)
    assert exact == 2
    assert near == []
    # dHash collision with DIFFERENT pixels must NOT fail the build:
    # identical hashes prove nothing, only pixels do.
    exact, near = v4.audit_cross_split(
        {"a": [(a, "P1")], "b": [(a, "P2")]},
        {"a": "train", "b": "holdout"}, 8)
    assert exact == 0
    assert near == [{"a": "a", "b": "b"}]
    # Near-similar (1 bit apart) warns once per video pair.
    exact, near = v4.audit_cross_split(
        {"a": [(0b0000, "P")], "b": [(0b0001, "Q")]},
        {"a": "train", "b": "val"}, 1)
    assert exact == 0
    assert near == [{"a": "a", "b": "b"}]
    # Same-split pairs are ignored entirely.
    exact, near = v4.audit_cross_split(
        {"a": [(a, "P")], "b": [(a, "P")]},
        {"a": "train", "b": "train"}, 8)
    assert (exact, near) == (0, [])


def test_resolve_weights(tmp_path):
    resolve_weights = _load_train_module().resolve_weights
    project = tmp_path / "runs"
    repo = tmp_path / "repo"
    repo.mkdir()
    # Explicit missing path fails loudly.
    with pytest.raises(SystemExit):
        resolve_weights(str(tmp_path / "nope.pt"), project, repo)
    have = tmp_path / "have.pt"
    have.write_text("x")
    w, _ = resolve_weights(str(have), project, repo)
    assert w == have.resolve()
    # V2 best.pt wins over the yolo11m fallback.
    v2best = project / "industrial-ppe-v2" / "weights" / "best.pt"
    v2best.parent.mkdir(parents=True)
    v2best.write_text("x")
    w, note = resolve_weights(None, project, repo)
    assert w == v2best and "V2" in note
    # Without V2: repo-root copy wins; without it, bare auto-download name.
    v2best.unlink()
    (repo / "yolo11m.pt").write_text("x")
    w, _ = resolve_weights(None, project, repo)
    assert w == repo / "yolo11m.pt"
    (repo / "yolo11m.pt").unlink()
    w, note = resolve_weights(None, project, repo)
    assert str(w) == "yolo11m.pt" and "network" in note


def test_resolve_weights_bare_name_downloads(tmp_path, monkeypatch):
    resolve_weights = _load_train_module().resolve_weights
    # CWD-relative existence must not leak in: run as if on a clean machine.
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "runs"
    repo = tmp_path / "repo"
    repo.mkdir()
    # Bare known name without a local file = auto-download identifier,
    # so the documented V4 command works on a clean machine.
    w, note = resolve_weights("yolo11m.pt", project, repo)
    assert str(w) == "yolo11m.pt" and "auto-download" in note
    w, _ = resolve_weights("./yolo11m.pt", project, repo)
    assert str(w) == "yolo11m.pt"
    # Anything path-shaped and missing still fails loudly.
    with pytest.raises(SystemExit):
        resolve_weights("weights/yolo11m.pt", project, repo)
    with pytest.raises(SystemExit):
        resolve_weights("yolo11m.ptt", project, repo)


def test_forbid_vacuous_guard_rejected():
    with pytest.raises(SystemExit):
        v4.forbid_pixel_shas(["/does/not/exist"])
    assert v4.forbid_pixel_shas([]) == set()


def test_single_video_split_rejected(tmp_path):
    footage = tmp_path / "footage"
    vdir = footage / "only_scene"
    vdir.mkdir(parents=True)
    v4.make_synthetic_video(vdir / "clip.mp4", seed=7, n_frames=4)
    (vdir / "LICENSE.json").write_text(json.dumps(
        {"origin": "self-recorded", "license": "owned",
         "source_url": None, "consent": True}))
    import argparse

    args = argparse.Namespace(footage=str(footage),
                              staging=str(tmp_path / "staging"),
                              extract_fps=2.0)
    v4.stage_extract(args)
    build_args = argparse.Namespace(
        footage=str(footage), staging=str(tmp_path / "staging"), seed=17,
        holdout_frac=0.34, val_frac=0.5, max_images=600, min_per_class=1,
        near_bits=8, allow_drafts=True, forbid_dirs=[],
        output=str(tmp_path / "out"), holdout=str(tmp_path / "hold"))
    # Labels exist so QA passes; the split gate must fail on 1 video.
    for img in (tmp_path / "staging" / "only_scene" / "images").glob("*.jpg"):
        (vdir / "labels").mkdir(exist_ok=True)
        (vdir / "labels" / f"{img.stem}.txt").write_text(
            "4 0.5 0.5 0.5 0.5\n")
    with pytest.raises(SystemExit):
        v4.stage_build(build_args)


def test_forbid_guard_rejects_old_pixels(tmp_path):
    import argparse

    footage = tmp_path / "footage"
    vids = []
    for name, seed in (("v1", 1), ("v2", 2), ("v3", 3)):
        vdir = footage / name
        vdir.mkdir(parents=True)
        v4.make_synthetic_video(vdir / "clip.mp4", seed=seed, n_frames=4)
        (vdir / "LICENSE.json").write_text(json.dumps(
            {"origin": "self-recorded", "license": "owned",
             "source_url": None, "consent": True}))
        vids.append(vdir)
    args = argparse.Namespace(footage=str(footage),
                              staging=str(tmp_path / "staging"),
                              extract_fps=5.0)
    v4.stage_extract(args)
    staged = sorted((tmp_path / "staging" / "v1" / "images").glob("*.jpg"))
    # Every frame gets a confirmed person label (min-per-class=1 needs person).
    for vdir in vids:
        lab = vdir / "labels"
        lab.mkdir(exist_ok=True)
        for img in (tmp_path / "staging" / vdir.name / "images").glob("*.jpg"):
            (lab / f"{img.stem}.txt").write_text("4 0.5 0.5 0.5 0.5\n")
    # Forbid dir holds a pixel copy of one staged frame -> build must fail.
    forbid = tmp_path / "old_data" / "images"
    forbid.mkdir(parents=True)
    import shutil

    shutil.copyfile(staged[0], forbid / staged[0].name)
    build_args = argparse.Namespace(
        footage=str(footage), staging=str(tmp_path / "staging"), seed=17,
        holdout_frac=0.0, val_frac=0.5, max_images=600, min_per_class=1,
        near_bits=8, allow_drafts=False, forbid_dirs=[str(tmp_path / "old_data")],
        output=str(tmp_path / "out"), holdout=str(tmp_path / "hold"))
    with pytest.raises(SystemExit):
        v4.stage_build(build_args)
