"""PPE identity continuity vs observation gaps (source timestamps)."""

import time
from concurrent.futures import Future

from app.domain.models import BoundingBox
from app.services.engine import Track
from app.services.independent_ppe import IndependentPpe


def person(tid: int, box=(0.2, 0.1, 0.5, 0.9), seen: float = 0.0) -> Track:
    return Track(
        id=tid,
        bbox=BoundingBox(*box),
        seen=seen,
        confidence=0.9,
        violations=[],
        ppe={},
        ppe_confidence={},
    )


def test_sparse_fps_keeps_version_while_track_stays_present():
    """1 FPS / 0.5 FPS must not churn versions (old 0.6s wall-gap bug)."""
    for fps in (0.5, 1, 2, 5, 10):
        ppe = IndependentPpe(absence_grace_seconds=0.15)
        dt = 1.0 / fps
        ppe.observe_tracks([person(1)], 0.0)
        v0 = ppe.versions[1]
        for step in range(1, 6):
            ppe.observe_tracks([person(1)], step * dt)
            assert ppe.versions[1] == v0, f"fps={fps} step={step}"


def test_missing_worker_clears_identity_after_grace():
    ppe = IndependentPpe(absence_grace_seconds=0.15)
    ppe.observe_tracks([person(1)], 0.0)
    v0 = ppe.versions[1]
    # Still within grace — version retained while absent.
    ppe.observe_tracks([], 0.10)
    assert 1 in ppe.versions and ppe.versions[1] == v0
    # Beyond grace — cleared.
    invalid = ppe.observe_tracks([], 0.20)
    assert 1 in invalid and 1 not in ppe.versions
    ppe.observe_tracks([person(1)], 0.30)
    assert ppe.versions[1] != v0


def test_crossing_workers_bump_both_versions():
    ppe = IndependentPpe()
    ppe.observe_tracks([person(1, (0.1, 0.1, 0.4, 0.9))], 0.0)
    ppe.observe_tracks([person(2, (0.6, 0.1, 0.9, 0.9))], 0.1)
    v1, v2 = ppe.versions[1], ppe.versions[2]
    invalid = ppe.observe_tracks(
        [
            person(1, (0.3, 0.1, 0.6, 0.9)),
            person(2, (0.35, 0.1, 0.65, 0.9)),
        ],
        0.2,
    )
    assert 1 in invalid and 2 in invalid
    assert ppe.versions[1] != v1 and ppe.versions[2] != v2


def test_take_rejects_packet_after_identity_break():
    class Sess:
        session_id = "s"
        generation = 1

        class metrics:
            @staticmethod
            def mark(_name):
                return None

            @staticmethod
            def observe(_name, _value):
                return None

    ppe = IndependentPpe(absence_grace_seconds=0.15)
    ppe.observe_tracks([person(1)], 0.0)
    packet_versions = dict(ppe.versions)
    future = Future()
    future.set_result([])

    class Packet:
        session_id = "s"
        generation = 1
        timestamp = 0.0
        submitted_at = time.monotonic()
        frame_id = 1
        versions = packet_versions
        tracks = [person(1)]
        detections = None

    ppe.pending = {Packet.frame_id: (future, Packet())}
    ppe.observe_tracks([], 0.5)
    ppe.observe_tracks([person(1)], 0.6)
    assert ppe.take(Sess(), now=0.7, source_freshness_limit=5.0, current_track_ids={1}) is None
    assert ppe.status == "identity_uncertain"


def test_late_ppe_accepted_when_sparse_but_continuous():
    class Sess:
        session_id = "s"
        generation = 1

        class metrics:
            @staticmethod
            def mark(_name):
                return None

            @staticmethod
            def observe(_name, _value):
                return None

    ppe = IndependentPpe(absence_grace_seconds=0.15)
    ppe.observe_tracks([person(1)], 0.0)
    packet_versions = dict(ppe.versions)
    future = Future()
    future.set_result([])

    class Packet:
        session_id = "s"
        generation = 1
        timestamp = 0.0
        submitted_at = time.monotonic()
        frame_id = 1
        versions = packet_versions
        tracks = [person(1)]
        detections = None

    ppe.pending = {Packet.frame_id: (future, Packet())}
    ppe.observe_tracks([person(1)], 1.0)
    ppe.observe_tracks([person(1)], 2.0)
    out = ppe.take(Sess(), now=2.1, source_freshness_limit=5.0, current_track_ids={1})
    assert out is not None
    assert ppe.status == "current"


def test_old_wall_clock_gap_no_longer_breaks_continuous_worker():
    """Regression: formerly `now - last_seen > 0.6` broke 1 FPS streams."""
    ppe = IndependentPpe(absence_grace_seconds=0.15)
    ppe.observe_tracks([person(1)], 0.0)
    v0 = ppe.versions[1]
    # Exactly the old failure mode: next observation 1.0s later, still present.
    ppe.observe_tracks([person(1)], 1.0)
    assert ppe.versions[1] == v0
