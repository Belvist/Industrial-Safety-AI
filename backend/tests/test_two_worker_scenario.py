"""Two-worker tracking scenario: crossing, disappearance, return.

Synthetic detections drive the REAL Tracker + TemporalEngine (no inference,
no randomness) with production timing (dt=0.2, confirm/clear 2.0s, 4/3).

Worker A (no_helmet, conf 0.8) moves; worker B (helmet, 0.85) crosses or
stands. A's violation vanishes frames 61-75, returns 76+.

Proven invariants (asserted):
- disappearance yields "lost", never "resolved" (no evidence != cleared);
- return yields exactly one new ACTIVE (no zero, no duplicates);
- B (compliant) never raises no_* incidents.

Known limitation (documented, NOT asserted as correct):
- when two workers cross, the IoU tracker can switch IDs, which emits a
  false RESOLVED on the old id plus a duplicate ACTIVE on the new one.
  test_crossing measures it; the xfail test below fails until fixed and
  flips to XPASS automatically. Synthetic only — real ID-switch rate needs
  labeled violation video (NOT TESTED).
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.domain.models import BoundingBox, Detection  # noqa: E402
from app.services.engine import (  # noqa: E402
    TemporalEngine,
    Tracker,
    assign_detections,
    cleared_assigned,
    evaluate_assigned,
)
from app.services.schemas import CameraConfig  # noqa: E402

CFG = CameraConfig(helmet=True, vest=True, mask=False, smoking=False)
DT = 0.2


def _person(x, conf=0.9):
    x = max(0.0, min(0.8, x))
    return Detection('person', conf, BoundingBox(x, 0.2, x + 0.2, 0.8))


def _head(x, cls, conf):
    cx = max(0.035, min(0.965, x + 0.1))
    return Detection(cls, conf, BoundingBox(cx - 0.035, 0.22, cx + 0.035, 0.32))


def run(b_crossing: bool):
    """Returns (actions, distinct_ids). actions: [(frame, action, (id, type))]."""
    tracker, engine = Tracker(), TemporalEngine()
    actions, ids = [], set()
    for frame in range(100):
        now = frame * DT
        dets = []
        if frame <= 60 or frame >= 76:
            if b_crossing:
                ax = 0.05 + 0.008 * (frame if frame <= 60 else 60 + (frame - 76))
            else:
                ax = 0.05 + 0.008 * (frame if frame <= 60 else 60 + (frame - 76))
                ax = min(ax, 0.35)
            dets += [_person(ax), _head(ax, 'no_helmet', 0.8)]
        bx = 0.75 - 0.007 * frame if b_crossing else 0.75
        dets += [_person(bx), _head(bx, 'helmet', 0.85)]
        tracks = tracker.update(dets, now)
        ids.update(t.id for t in tracks)
        assigns, _ = assign_detections(tracks, dets, CFG)
        observed = evaluate_assigned(tracks, assigns, CFG)
        cleared = cleared_assigned(tracks, assigns, CFG)
        for action, key, _ in engine.update(observed, now, 2.0, 2.0, 4, 3, cleared):
            actions.append((frame, action, key))
    return actions, ids


def _of(actions, action, vtype='no_helmet'):
    return [(f, k) for f, a, k in actions if a == action and k[1] == vtype]


def test_disappearance_yields_lost_never_resolved():
    actions, _ids = run(b_crossing=False)
    assert _of(actions, 'resolved') == []
    lost = _of(actions, 'lost')
    assert len(lost) == 1 and 65 <= lost[0][0] <= 75


def test_return_yields_exactly_one_new_active():
    actions, _ids = run(b_crossing=False)
    actives = _of(actions, 'active')
    assert len(actives) == 2
    first, second = actives
    assert 8 <= first[0] <= 14
    assert 80 <= second[0] <= 95
    assert first[1][0] != second[1][0]  # new track id after expiry: documented
    assert _of(actions, 'resolved') == []


def test_compliant_worker_never_raises():
    # B wears a helmet the whole run: every no_helmet ACTIVE must belong to
    # A's two presence intervals (phase 1 + return). Helmet-class incidents
    # must never appear (helmet is not a violation type).
    actions, _ids = run(b_crossing=False)
    assert len(_of(actions, 'active')) == 2  # A-phase-1 + A-return; B contributes zero
    assert all(k[1] == 'no_helmet' for _, _, k in actions)


def test_crossing_keeps_single_incident():
    """Crossing workers keep their IDs: no false RESOLVED, no duplicate ACTIVE.

    Fixed by velocity-predicted association (was xfail: switch at f=57 with
    false RESOLVED + duplicate ACTIVE). If this regresses, the xfail must
    come back instead of silently accepting the bug.
    """
    actions, _ids = run(b_crossing=True)
    assert _of(actions, 'resolved') == []
    assert len(_of(actions, 'active')) == 2


def test_predicted_boxes_never_crash_at_edges():
    """Fast track near the frame edge: velocity extrapolation must clamp to
    [0,1] instead of raising (regression: whole-pipeline crash on real video).
    """
    tracker = Tracker()
    fast = Detection('person', 0.9, BoundingBox(0.75, 0.2, 0.95, 0.8))
    tracker.update([fast], 0.0)
    tracker.update([Detection('person', 0.9, BoundingBox(0.80, 0.2, 1.0, 0.8))], 0.2)
    # large dt: prediction would fly far outside without clamping
    tracks = tracker.update(
        [Detection('person', 0.9, BoundingBox(0.0, 0.2, 0.2, 0.8))], 1.5)
    assert all(0.0 <= v <= 1.0 for t in tracks
               for v in (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2))


@pytest.mark.parametrize('box', [
    (1.0, 0.2, 1.0, 0.8),    # audit case: x1 == x2 == 1.0
    (0.0, 0.2, 0.0, 0.8),    # x1 == x2 == 0.0
    (0.2, 0.0, 0.8, 0.0),    # y1 == y2 == 0.0
    (0.2, 1.0, 0.8, 1.0),    # y1 == y2 == 1.0
    (1.5, 1.5, 2.0, 2.0),    # fully outside
    (0.8, 0.8, 0.2, 0.2),    # inverted
    (-0.5, -0.5, -0.1, -0.1),  # fully outside, negative
])
def test_clamp_predicted_all_edges(box):
    from app.services.engine import clamp_predicted

    out = clamp_predicted(*box)
    assert 0.0 <= out.x1 < out.x2 <= 1.0
    assert 0.0 <= out.y1 < out.y2 <= 1.0


def test_crossing_outcome_is_measured():
    """Documents fixed crossing behavior as data: 2 ACTIVE, 1 lost, 0 resolved."""
    actions, ids = run(b_crossing=True)
    by_action = Counter(a for _, a, _ in actions)
    assert by_action['active'] == 2
    assert by_action['lost'] == 1
    assert by_action.get('resolved', 0) == 0
    assert sorted(ids) == [1, 2, 3]
