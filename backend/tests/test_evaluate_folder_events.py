"""Track-aware incident matching for evaluate_folder --events.

Core regression: an alarm on the WRONG person must never score as a match,
even with identical type and overlapping time.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location(
    "evaluate_folder", ROOT / "backend" / "scripts" / "evaluate_folder.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def det(typ, start, end, boxes=None, worker=1):
    return {"type": typ, "start_frame": start, "end_frame": end,
            "worker": worker, "boxes": boxes or {}}


def gt(typ, start, end, worker="w1", boxes=None):
    e = {"type": typ, "start_frame": start, "end_frame": end, "worker": worker}
    if boxes is not None:
        e["boxes"] = boxes
    return e


BOX_A = [0.1, 0.1, 0.4, 0.6]
BOX_B = [0.6, 0.1, 0.9, 0.6]  # far from A: IoU 0


def test_match_same_worker():
    d = [det("no_helmet", 10, 50, {10: BOX_A, 30: BOX_A, 50: BOX_A})]
    g = [gt("no_helmet", 12, 48, "w1", {"12": BOX_A, "30": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["missed"] == [] and r["false_alarms"] == []
    assert r["latencies"] == [-2]
    assert r["worker_mismatch"] == [] and r["unverified"] == []


def test_wrong_worker_never_matches():
    d = [det("no_helmet", 10, 50, {10: BOX_B, 30: BOX_B})]
    g = [gt("no_helmet", 12, 48, "w1", {"12": BOX_A, "30": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == []
    assert r["missed"] == [0]
    assert r["false_alarms"] == [0]
    assert r["worker_mismatch"] == [0]


def test_legacy_gt_without_boxes_time_only_unverified():
    d = [det("no_vest", 5, 20, {5: BOX_A})]
    g = [gt("no_vest", 6, 18, "w1")]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["unverified"] == [0]
    assert r["worker_mismatch"] == []


def test_no_common_box_frames_falls_back_unverified():
    d = [det("no_helmet", 10, 50, {40: BOX_B})]
    g = [gt("no_helmet", 12, 48, "w1", {"12": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["unverified"] == [0]


def test_one_to_one_and_type_mismatch():
    d = [det("no_helmet", 10, 20, {10: BOX_A}),
         det("no_helmet", 12, 22, {12: BOX_A}),
         det("no_vest", 10, 20, {10: BOX_A})]
    g = [gt("no_helmet", 11, 19, "w1", {"11": BOX_A})]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    assert r["false_alarms"] == [1, 2]
    assert r["missed"] == []


def test_strict_metric_excludes_time_only():
    d = [det("no_vest", 5, 20, {5: BOX_A})]
    g = [gt("no_vest", 6, 18, "w1")]
    r = ev.match_incidents(d, g)
    assert r["matched"] == [(0, 0)]
    unverified = set(r["unverified"])
    verified = [m for m in r["matched"] if m[0] not in unverified]
    assert verified == []  # time-only must NOT count as worker-verified
    d2 = [det("no_vest", 5, 20, {6: BOX_A, 10: BOX_A})]
    g2 = [gt("no_vest", 6, 18, "w1", {"6": BOX_A, "10": BOX_A})]
    r2 = ev.match_incidents(d2, g2)
    verified2 = [m for m in r2["matched"] if m[0] not in set(r2["unverified"])]
    assert verified2 == [(0, 0)]


def test_spatial_agreement_majority():
    agree, ver = ev._spatial_agreement(
        {1: BOX_A, 2: BOX_A, 3: BOX_B}, {1: BOX_A, 2: BOX_A, 3: BOX_A}, 1, 3)
    assert (agree, ver) == (True, True)
    agree, ver = ev._spatial_agreement(
        {1: BOX_B, 2: BOX_B, 3: BOX_A}, {1: BOX_A, 2: BOX_A, 3: BOX_A}, 1, 3)
    assert (agree, ver) == (False, True)
    agree, ver = ev._spatial_agreement({99: BOX_A}, {1: BOX_A}, 1, 50)
    assert (agree, ver) == (False, False)
