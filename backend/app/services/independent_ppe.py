"""Bounded, source-associated PPE results; motion never waits for this branch."""

import copy
import math
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
from uuid import uuid4

import cv2

from app.services.engine import TemporalEngine, iou


def observation_quality(frame, tracks):
    h, w = frame.shape[:2]
    scale = min(640 / w, 640 / h)
    small = cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luminance = float(gray.mean())
    reasons = []
    if sharpness < 12:
        reasons.append("blurred_frame")
    if luminance < 25:
        reasons.append("dark_frame")
    people = {}
    for track in tracks:
        height = (track.bbox.y2 - track.bbox.y1) * h * scale
        flags = reasons + (["person_too_small"] if height < 72 else [])
        people[track.id] = {
            "person_model_pixels": round(height, 1),
            "ppe_observable": not flags,
            "reasons": flags,
        }
    return {
        "sharpness": round(sharpness, 2),
        "luminance": round(luminance, 1),
        "minimum_person_model_pixels": 72,
        "reasons": reasons,
        "tracks": people,
        "note": "Uncalibrated conservative admission heuristics; a hold does not prove the model cannot detect PPE.",
        "guard_calibrated": False,
        "failure_cause_proven": False,
    }


@dataclass
class PpePacket:
    frame: object
    tracks: list
    versions: dict
    quality: dict
    timestamp: float
    captured: float
    frame_id: int
    session_id: str
    generation: int
    detections: list | None = None
    request_id: str = field(default_factory=lambda: uuid4().hex)


class BoundedPpeExecutor:
    """Bounded FIFO admission across cameras, including old generations.

    At most `limit` jobs run; at most eight wait (the configured-camera cap).
    A busy camera cannot repeatedly steal every newly released worker slot.
    Cancelled queued items remain inside the bounded queue until drained, so
    rapid stop/restart cannot grow an unbounded ThreadPoolExecutor work queue.
    """

    def __init__(self, limit, queue_capacity=8):
        if limit < 1 or queue_capacity < 1:
            raise ValueError("PPE worker and queue limits must be positive")
        self.queue = queue.Queue(maxsize=queue_capacity)
        self.lock = threading.Lock()
        self.closed = False
        self.stopping = threading.Event()
        self.workers = [
            threading.Thread(target=self._run, name=f"independent-ppe-{i}", daemon=True)
            for i in range(limit)
        ]
        for worker in self.workers:
            worker.start()

    def try_submit(self, work):
        with self.lock:
            if self.closed:
                return None
            future = Future()
            try:
                self.queue.put_nowait((future, work))
            except queue.Full:
                return None
            return future

    def _run(self):
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                future, work = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                if not future.set_running_or_notify_cancel():
                    continue
                try:
                    result = work()
                except BaseException as error:
                    future.set_exception(error)
                else:
                    future.set_result(result)
            finally:
                self.queue.task_done()

    def shutdown(self, wait=True, cancel_futures=True):
        with self.lock:
            self.closed = True
            self.stopping.set()
            if cancel_futures:
                while True:
                    try:
                        future, _ = self.queue.get_nowait()
                    except queue.Empty:
                        break
                    future.cancel()
                    self.queue.task_done()
        if wait:
            for worker in self.workers:
                worker.join()


class IndependentPpe:
    def __init__(self):
        self.engine = TemporalEngine()
        # Insertion order is strictly increasing source time. Completed replies
        # wait here for their predecessors, never for motion processing.
        self.pending = {}
        self.retired = {}
        self.limit = 1
        self.versions = {}
        self.last_seen = {}
        self.serial = 0
        self.values = {}
        self.last_submitted = None
        self.next_due = None
        self.last_submission_frame_id = None
        self.last_result = None
        self.last_frame_id = None
        self.status = "unknown"
        self.error = None

    def reset(self):
        for future, _ in list(self.pending.values()) + list(self.retired.values()):
            future.cancel()
        self.__init__()

    def observe_tracks(self, tracks, now):
        """An occlusion/crossing invalidates PPE identity continuity, not motion IDs."""
        present = {t.id for t in tracks}
        invalid = set(self.versions) - present
        for t in tracks:
            ambiguous = any(o.id != t.id and iou(t.bbox, o.bbox) > 0.2 for o in tracks)
            broken = now - self.last_seen.get(t.id, now) > 0.6
            if t.id not in self.versions or ambiguous or broken:
                if t.id in self.versions:
                    invalid.add(t.id)
                self.serial += 1
                self.versions[t.id] = self.serial
            self.last_seen[t.id] = now
        for tid in list(self.versions):
            if tid not in present:
                self.versions.pop(tid, None)
                self.last_seen.pop(tid, None)
        for tid in invalid:
            self.values.pop(tid, None)
        return invalid

    def take(self, session, now, max_age):
        """Return at most one valid result per ordered motion commit.

        Expired heads do not block newer completed results. An expired running
        Future is quarantined, still counting against admission until finished;
        its late answer is never applied. At most one packet is applied per
        commit so incident state/snapshot remains tied to that exact packet.
        """
        for key, (future, _) in list(self.retired.items()):
            if future.done():
                self.retired.pop(key)
                session.metrics.mark("ppe_late_discarded")
        while self.pending:
            key = next(iter(self.pending))
            future, packet = self.pending[key]
            invalid = (
                packet.session_id != session.session_id or packet.generation != session.generation
            )
            expired = now - packet.timestamp > max_age
            if invalid or expired:
                self.pending.pop(key)
                future.cancel()
                if not future.done():
                    self.retired[key] = (future, packet)
                session.metrics.mark("ppe_invalid_generation" if invalid else "ppe_stale")
                self.status = "stale"
                continue
            if not future.done():
                return None
            self.pending.pop(key)
            try:
                packet.detections = future.result()
            except Exception:
                self.status, self.error = "unavailable", "PPE inference unavailable"
                session.metrics.mark("ppe_failures")
                continue
            if self.last_result is not None and packet.timestamp <= self.last_result:
                session.metrics.mark("ppe_out_of_order_discarded")
                continue
            packet.tracks = [
                t for t in packet.tracks if self.versions.get(t.id) == packet.versions[t.id]
            ]
            if not packet.tracks:
                self.status = "identity_uncertain"
                session.metrics.mark("ppe_identity_rejected")
                continue
            self.last_frame_id = packet.frame_id
            self.last_result = packet.timestamp
            self.status, self.error = "current", None
            session.metrics.mark("ppe_applied")
            session.metrics.observe("ppe_applied_age_ms", max(0, now - packet.timestamp) * 1000)
            return packet
        return None

    def submit(
        self,
        session,
        frame,
        tracks,
        quality,
        now,
        captured,
        pool,
        predict,
        interval,
        limit=1,
        max_age=2.5,
    ):
        self.limit = limit
        if len(self.pending) + len(self.retired) >= limit or not tracks:
            return False
        frame_id = session.sequence + 1
        if (
            self.last_submission_frame_id is not None
            and frame_id <= self.last_submission_frame_id
            or self.last_submitted is not None
            and now <= self.last_submitted
        ):
            return False
        if self.next_due is not None and now + 1e-6 < self.next_due:
            return False
        if not any(quality["tracks"][t.id]["ppe_observable"] for t in tracks):
            self.status = "insufficient_image_quality"
            session.metrics.mark("ppe_quality_skips")
            return False
        packet = PpePacket(
            frame.copy(),
            copy.deepcopy(tracks),
            dict(self.versions),
            quality,
            now,
            captured,
            frame_id,
            session.session_id,
            session.generation,
        )

        def work():
            if (
                session.stop.is_set()
                or session.session_id != packet.session_id
                or session.generation != packet.generation
            ):
                raise RuntimeError("Cancelled source generation")
            if (
                session.last_timestamp is not None
                and session.last_timestamp - packet.timestamp > max_age
            ):
                raise RuntimeError("PPE frame expired before inference")
            return predict(packet.frame, packet.tracks)

        future = pool.try_submit(work)
        if future is None:
            session.metrics.mark("ppe_backpressure")
            return False
        self.pending[packet.frame_id] = (future, packet)
        self.last_submitted = now
        self.last_submission_frame_id = packet.frame_id
        due = self.next_due if self.next_due is not None else now
        self.next_due = due + max(1, math.floor((now - due + 1e-6) / interval) + 1) * interval
        session.metrics.mark("ppe_submitted")
        if self.last_result is None and self.error is None:
            self.status = "pending"
        return True

    def describe(self, now, max_age):
        age = max(0, now - self.last_result) if self.last_result is not None else None
        packets = list(self.pending.values()) + list(self.retired.values())
        return {
            "independent_motion": True,
            "state": self.status,
            "age_seconds": round(age, 3) if age is not None else None,
            "maximum_age_seconds": max_age,
            "pending": bool(packets),
            "pending_count": len(self.pending),
            "retired_count": len(self.retired),
            "in_flight": sum(future.running() for future, _ in packets),
            "queued": sum(not future.done() and not future.running() for future, _ in packets),
            "completed_waiting": sum(future.done() for future, _ in self.pending.values()),
            "maximum_in_flight_per_camera": self.limit,
            "oldest_request_age_seconds": round(
                max(0, now - min(p.timestamp for _, p in packets)), 3
            )
            if packets
            else None,
            "source_timestamp": self.last_result,
            "source_frame_id": self.last_frame_id,
            "error": self.error,
        }
