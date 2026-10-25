"""Deterministic bounded PPE ordering; not inference or accelerator accuracy proof."""

import threading
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np

from app.domain.models import BoundingBox
from app.services.engine import Track
from app.services.independent_ppe import BoundedPpeExecutor, IndependentPpe
from app.services.metrics import Metrics


class Deferred:
    def __init__(self):
        self.jobs = []

    def try_submit(self, fn):
        f = Future()
        self.jobs.append((f, fn))
        return f


def scene():
    session = SimpleNamespace(
        sequence=0,
        session_id="s",
        generation=0,
        stop=threading.Event(),
        metrics=Metrics(),
        last_timestamp=0,
    )
    ppe = IndependentPpe()
    track = Track(1, BoundingBox(0.1, 0.1, 0.4, 0.9), 0, 0.99)
    frame = np.zeros((32, 32, 3), np.uint8)
    quality = {"tracks": {1: {"ppe_observable": True}}}
    return ppe, session, track, frame, quality


def submit(ppe, s, t, frame, q, pool, now, limit=4):
    ppe.observe_tracks([t], now)
    s.last_timestamp = now
    accepted = ppe.submit(s, frame, [t], q, now, now, pool, lambda *_: [], 0.2, limit=limit)
    s.sequence += 1
    return accepted


def test_slow_rpc_can_apply_only_while_source_remains_fresh(monkeypatch):
    import app.services.independent_ppe as ppe_module

    monkeypatch.setattr(ppe_module.time, "monotonic", lambda: 100.0)
    for rpc_delay in (1.0, 2.5, 5.0, 8.0):
        ppe, s, t, frame, q = scene()
        pool = Deferred()
        assert submit(ppe, s, t, frame, q, pool, 10.0)
        future, packet = next(iter(ppe.pending.values()))
        packet.submitted_at -= rpc_delay
        future.set_result([])
        accepted = ppe.take(s, 11.0, 2.5, 8.0, {t.id})
        assert accepted is packet
        assert s.metrics.snapshot()["stages"]["rpc_wait_ms"]["p50"] >= rpc_delay * 1000
        assert s.metrics.snapshot()["stages"]["ppe_applied_age_ms"]["p50"] == 1000


def test_rpc_deadline_rejects_after_eight_seconds():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    assert submit(ppe, s, t, frame, q, pool, 10.0)
    future, packet = next(iter(ppe.pending.values()))
    packet.submitted_at -= 12.0
    assert ppe.take(s, 10.1, 2.5, 8.0, {t.id}) is None
    assert packet.frame_id not in ppe.pending
    assert future.cancelled()
    assert s.metrics.snapshot()["rates"]["ppe_rejected_stale"]["total"] == 1


def test_slow_reply_for_stale_source_is_rejected_even_within_rpc_budget():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    assert submit(ppe, s, t, frame, q, pool, 10.0)
    future, packet = next(iter(ppe.pending.values()))
    packet.submitted_at -= 1.0
    future.set_result([])
    assert ppe.take(s, 13.0, 2.5, 8.0, {t.id}) is None
    assert ppe.last_result is None
    assert s.metrics.snapshot()["rates"]["ppe_rejected_stale"]["total"] == 1


def test_result_for_disappeared_track_is_rejected():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    assert submit(ppe, s, t, frame, q, pool, 10.0)
    pool.jobs[0][0].set_result([])
    assert ppe.take(s, 10.1, 2.5, 8.0, set()) is None
    assert ppe.last_result is None
    assert s.metrics.snapshot()["rates"]["ppe_identity_rejected"]["total"] == 1


def test_old_generation_and_replaced_track_identity_are_not_applied():
    for mode in ("restart", "new_id"):
        ppe, s, t, frame, q = scene()
        pool = Deferred()
        assert submit(ppe, s, t, frame, q, pool, 10.0)
        future = pool.jobs[0][0]
        if mode == "restart":
            s.generation += 1
        else:
            ppe.observe_tracks([], 10.2)
            replacement = Track(2, t.bbox, 0.2, 0.99)
            ppe.observe_tracks([replacement], 10.4)
        future.set_result([])
        current = {2} if mode == "new_id" else {t.id}
        assert ppe.take(s, 10.5, 2.5, 8.0, current) is None
        assert ppe.last_result is None


def test_rpc_timeout_rejects_and_quarantines_pending_reply():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    assert submit(ppe, s, t, frame, q, pool, 10.0)
    future, packet = next(iter(ppe.pending.values()))
    packet.submitted_at -= 9.0
    assert ppe.take(s, 10.1, 2.5, 8.0, {t.id}) is None
    assert packet.frame_id not in ppe.pending
    assert future.cancelled()
    assert s.metrics.snapshot()["rates"]["ppe_rejected_stale"]["total"] == 1


def test_four_out_of_order_results_apply_in_original_order_with_unique_identity():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    for now in [0, 0.2, 0.4, 0.6]:
        assert submit(ppe, s, t, frame, q, pool, now)
    assert len(ppe.pending) == 4
    packets = [packet for _, packet in ppe.pending.values()]
    assert len({p.request_id for p in packets}) == 4
    assert [p.frame_id for p in packets] == [1, 2, 3, 4]
    assert not submit(ppe, s, t, frame, q, pool, 0.8)
    # Newer replies are buffered. They cannot overwrite an earlier source state.
    for f, _ in reversed(pool.jobs[1:]):
        f.set_result([])
    def take(now, source_limit=2.5, rpc_budget=2.5):
        return ppe.take(s, now, source_limit, rpc_budget, {t.id})

    assert take(0.8) is None
    pool.jobs[0][0].set_result([])
    assert [take(0.8).timestamp for _ in range(4)] == [0, 0.2, 0.4, 0.6]
    assert ppe.last_result == 0.6 and not ppe.pending
    assert s.metrics.snapshot()["rates"]["ppe_applied"]["total"] == 4


def test_expired_running_head_does_not_block_or_overwrite_newer_reply():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    assert submit(ppe, s, t, frame, q, pool, 0, 2)
    assert submit(ppe, s, t, frame, q, pool, 0.2, 2)
    old = pool.jobs[0][0]
    assert old.set_running_or_notify_cancel()
    pool.jobs[1][0].set_result([])
    # Make the head genuinely old in wall-clock too: a reply may only be
    # discarded when BOTH source age and wall age exceed max_age.
    for _, packet in ppe.pending.values():
        packet.submitted_at -= 5.0
    packet = ppe.take(s, 1.1, 1, 10.0, {t.id})
    assert packet.timestamp == 0.2
    assert len(ppe.retired) == 1 and ppe.last_result == 0.2
    # Quarantined work still occupies a per-camera slot.
    assert submit(ppe, s, t, frame, q, pool, 1.2, 2)
    assert not submit(ppe, s, t, frame, q, pool, 1.4, 2)
    old.set_result([])
    assert ppe.take(s, 1.4, 1, 2.5, {t.id}) is None
    assert ppe.last_result == 0.2
    assert s.metrics.snapshot()["rates"]["ppe_late_discarded"]["total"] == 1


def test_reset_invalidates_snapshots_and_cancels_waiting_work():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    submit(ppe, s, t, frame, q, pool, 0)
    old = pool.jobs[0][0]
    ppe.reset()
    s.generation += 1
    assert old.cancelled() and ppe.take(s, 0.2, 2.5, 2.5, set()) is None
    assert ppe.last_result is None


def test_global_running_limit_and_fifo_survive_camera_reset():
    ppe, s, t, frame, q = scene()
    pool = BoundedPpeExecutor(1)
    entered, release = threading.Event(), threading.Event()
    active = 0
    peak = 0

    def work(*_):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        release.wait(2)
        active -= 1
        return []

    try:
        ppe.observe_tracks([t], 0)
        assert ppe.submit(s, frame, [t], q, 0, 0, pool, work, 0.2)
        future = next(iter(ppe.pending.values()))[0]
        assert entered.wait(0.5)
        ppe.reset()
        s.generation += 1
        s.sequence += 1
        ppe.observe_tracks([t], 0.2)
        assert ppe.submit(s, frame, [t], q, 0.2, 0.2, pool, work, 0.2)
        new = next(iter(ppe.pending.values()))[0]
        assert not new.running()
        assert ppe.describe(0.2, 2.5, 5.0, 2.5)["queued"] == 1
        release.set()
        future.result(timeout=1)
        new.result(timeout=1)
        assert peak == 1
        assert ppe.take(s, 0.4, 2.5, 2.5, {t.id}).generation == 1
    finally:
        release.set()
        pool.shutdown()


def test_global_queue_is_bounded_and_fifo_across_owners():
    pool = BoundedPpeExecutor(1, queue_capacity=3)
    entered, release = threading.Event(), threading.Event()
    order = []

    def held():
        entered.set()
        release.wait(2)

    try:
        old = pool.try_submit(held)
        assert entered.wait(0.5)
        futures = [pool.try_submit(lambda value=i: order.append(value)) for i in range(3)]
        assert all(futures)
        assert pool.try_submit(lambda: None) is None
        assert pool.queue.qsize() == 3
        release.set()
        old.result(timeout=1)
        for future in futures:
            future.result(timeout=1)
        assert order == [0, 1, 2]
    finally:
        release.set()
        pool.shutdown()


def test_changed_id_or_ambiguous_crossing_cannot_receive_old_ppe():
    for mode in ["id_change", "overlap"]:
        ppe, s, t, frame, q = scene()
        pool = Deferred()
        submit(ppe, s, t, frame, q, pool, 0)
        other = Track(2, t.bbox, 0, 0.9)
        ppe.observe_tracks([other] if mode == "id_change" else [t, other], 0.2)
        pool.jobs[0][0].set_result([])
        assert ppe.take(s, 0.2, 2.5, 2.5, {t.id}) is None
        assert ppe.last_result is None


def test_duplicate_source_time_or_frame_cannot_create_second_request():
    ppe, s, t, frame, q = scene()
    pool = Deferred()
    assert submit(ppe, s, t, frame, q, pool, 0)
    assert not submit(ppe, s, t, frame, q, pool, 0)
    s.sequence = 0
    assert not submit(ppe, s, t, frame, q, pool, 0.2)
    assert len(pool.jobs) == 1
