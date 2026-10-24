"""Bounded, source-associated PPE results; motion never waits for this branch."""

import copy
from dataclasses import dataclass

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
        "note": "Heuristic observation checks, not a calibrated accuracy estimate.",
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


class IndependentPpe:
    def __init__(self):
        self.engine = TemporalEngine()
        self.pending = None
        self.versions = {}
        self.last_seen = {}
        self.serial = 0
        self.values = {}
        self.last_submitted = None
        self.last_result = None
        self.last_frame_id = None
        self.status = "unknown"
        self.error = None

    def reset(self):
        if self.pending:
            self.pending[0].cancel()
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
        if not self.pending or not self.pending[0].done():
            return None
        future, packet = self.pending
        self.pending = None
        try:
            packet.detections = future.result()
        except Exception:
            self.status, self.error = "unavailable", "PPE inference unavailable"
            session.metrics.mark("ppe_failures")
            return None
        if (
            packet.session_id != session.session_id
            or packet.generation != session.generation
            or now - packet.timestamp > max_age
        ):
            self.status = "stale"
            session.metrics.mark("ppe_stale")
            return None
        packet.tracks = [
            t for t in packet.tracks if self.versions.get(t.id) == packet.versions[t.id]
        ]
        if not packet.tracks:
            self.status = "identity_uncertain"
            session.metrics.mark("ppe_identity_rejected")
            return None
        self.last_frame_id = packet.frame_id
        self.last_result = packet.timestamp
        self.status, self.error = "current", None
        return packet

    def submit(self, session, frame, tracks, quality, now, captured, pool, predict, interval):
        if self.pending is not None or not tracks:
            return
        if self.last_submitted is not None and now - self.last_submitted < interval:
            return
        if not any(quality["tracks"][t.id]["ppe_observable"] for t in tracks):
            self.status = "insufficient_image_quality"
            session.metrics.mark("ppe_quality_skips")
            return
        packet = PpePacket(
            frame.copy(),
            copy.deepcopy(tracks),
            dict(self.versions),
            quality,
            now,
            captured,
            session.sequence + 1,
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
            return predict(packet.frame, packet.tracks)

        self.pending = (pool.submit(work), packet)
        self.last_submitted = now
        if self.last_result is None and self.error is None:
            self.status = "pending"

    def describe(self, now, max_age):
        age = max(0, now - self.last_result) if self.last_result is not None else None
        return {
            "independent_motion": True,
            "state": self.status,
            "age_seconds": round(age, 3) if age is not None else None,
            "maximum_age_seconds": max_age,
            "pending": self.pending is not None,
            "source_timestamp": self.last_result,
            "source_frame_id": self.last_frame_id,
            "error": self.error,
        }
