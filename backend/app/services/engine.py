"""Deterministic tracking and safety state machine; independent of inference hardware."""

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum

from app.domain.models import BoundingBox, Detection
from app.services.schemas import CameraConfig


def iou(a: BoundingBox, b: BoundingBox) -> float:
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
        0, min(a.y2, b.y2) - max(a.y1, b.y1)
    )
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection
    return intersection / union if union else 0


def contains(polygon, point) -> bool:
    x, y = point
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if (
            abs(cross) < 1e-9
            and min(ax, bx) <= x <= max(ax, bx)
            and min(ay, by) <= y <= max(ay, by)
        ):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


@dataclass
class Track:
    id: int
    bbox: BoundingBox
    seen: float
    confidence: float
    violations: list[str] = field(default_factory=list)
    zone_name: str | None = None
    ppe: dict = field(default_factory=dict)
    ppe_confidence: dict = field(default_factory=dict)
    ppe_age_seconds: float = 0.0
    # Constant-velocity estimate (normalized units/sec), updated on match.
    # Used ONLY to predict the box before association so crossing workers
    # keep their IDs; never leaks label information into tracking.
    vx: float = 0.0
    vy: float = 0.0


class TrackerProvider(ABC):
    """Tracking contract: stable worker IDs across frames within a session."""

    @abstractmethod
    def update(self, detections: list[Detection], now: float) -> list[Track]:
        raise NotImplementedError


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def clamp_predicted(x1: float, y1: float, x2: float, y2: float):
    """Clamp a velocity-predicted box into a VALID [0,1] box, always.

    Association-only helper: must never raise, even when the whole box flew
    past an edge (x1 == x2 == 1.0) or inverted. Degenerate predictions
    collapse to an epsilon box at the nearest edge — matchable to nothing,
    but never a pipeline crash.
    """
    eps = 1e-3
    x1, y1, x2, y2 = _clamp01(x1), _clamp01(y1), _clamp01(x2), _clamp01(y2)
    if x2 - x1 < eps:
        cx = min(max((x1 + x2) / 2, eps / 2), 1 - eps / 2)
        x1, x2 = cx - eps / 2, cx + eps / 2
    if y2 - y1 < eps:
        cy = min(max((y1 + y2) / 2, eps / 2), 1 - eps / 2)
        y1, y2 = cy - eps / 2, cy + eps / 2
    return BoundingBox(x1, y1, x2, y2)


class SimpleIoUTracker(TrackerProvider):
    """Greedy IoU matching with a center-distance second stage.

    Stage 1 matches person detections to recent tracks by IoU. Stage 2 keeps
    the ID when IoU briefly dips (fast move, partial occlusion) but the
    center stays within half the track box size. No appearance/biometric data.
    """

    def __init__(self, iou_threshold: float = 0.15, max_age: float = 3.0):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, detections: list[Detection], now: float) -> list[Track]:
        self.tracks = {k: v for k, v in self.tracks.items() if now - v.seen <= self.max_age}
        # Only confident persons become workers; weak person boxes stay
        # visible diagnostics without spawning Worker tracks.
        people = [d for d in detections if d.class_name == "person" and not d.weak]
        # Predict current boxes by constant velocity so that two crossing
        # workers associate with their own trajectories, not each other.
        # Association stays strictly one-to-one (matched/used sets below).
        predicted: dict[int, BoundingBox] = {}
        for key, track in self.tracks.items():
            age = max(0.0, now - track.seen)
            predicted[key] = clamp_predicted(
                track.bbox.x1 + track.vx * age, track.bbox.y1 + track.vy * age,
                track.bbox.x2 + track.vx * age, track.bbox.y2 + track.vy * age)
        matched: dict[int, int] = {}
        used: set[int] = set()
        candidates = []
        for index, person in enumerate(people):
            for key in self.tracks:
                overlap = iou(person.bbox, predicted[key])
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, index, key))
        for _, index, key in sorted(candidates, reverse=True):
            if index not in matched and key not in used:
                matched[index] = key
                used.add(key)
        # Stage 2: center distance for the still-unmatched, also predicted.
        for index, person in enumerate(people):
            if index in matched:
                continue
            cx = (person.bbox.x1 + person.bbox.x2) / 2
            cy = (person.bbox.y1 + person.bbox.y2) / 2
            best = None
            for key in self.tracks:
                if key in used:
                    continue
                p = predicted[key]
                gate = 0.6 * max(p.x2 - p.x1, p.y2 - p.y1)
                dist = math.hypot(cx - (p.x1 + p.x2) / 2, cy - (p.y1 + p.y2) / 2)
                if dist <= gate and (best is None or dist < best[0]):
                    best = (dist, key)
            if best is not None:
                matched[index] = best[1]
                used.add(best[1])
        current = []
        for index, person in enumerate(people):
            key = matched.get(index)
            if key is None:
                key = self.next_id
                self.next_id += 1
            prev = self.tracks.get(key)
            if prev is None:
                track = Track(key, person.bbox, now, person.confidence)
            else:
                dt = now - prev.seen
                if dt > 1e-6:
                    nvx = ((person.bbox.x1 + person.bbox.x2) / 2
                           - (prev.bbox.x1 + prev.bbox.x2) / 2) / dt
                    nvy = ((person.bbox.y1 + person.bbox.y2) / 2
                           - (prev.bbox.y1 + prev.bbox.y2) / 2) / dt
                    vx = 0.6 * prev.vx + 0.4 * nvx
                    vy = 0.6 * prev.vy + 0.4 * nvy
                else:
                    vx, vy = prev.vx, prev.vy
                track = Track(key, person.bbox, now, person.confidence,
                              vx=vx, vy=vy)
            self.tracks[key] = track
            current.append(track)
        return current


# Historic name kept for runtime/tests; the provider boundary is TrackerProvider.
Tracker = SimpleIoUTracker


def _overlap_area(box: BoundingBox, person: BoundingBox, margin: float = 0.05) -> float:
    x1 = max(box.x1, person.x1 - margin * (person.x2 - person.x1))
    y1 = max(box.y1, person.y1 - margin * (person.y2 - person.y1))
    x2 = min(box.x2, person.x2 + margin * (person.x2 - person.x1))
    y2 = min(box.y2, person.y2 + margin * (person.y2 - person.y1))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def assign_detections(
    tracks: list[Track], detections: list[Detection], config: CameraConfig
) -> tuple[dict[int, dict[str, float]], dict[int, int | None]]:
    """Map each detection to at most one worker track.

    Returns (assignments per track, owner per detection index or None).
    A detection must overlap its worker's box: a vest lying beside a person
    (zero overlap) stays unassigned instead of becoming that worker's PPE.
    """
    assignments = {t.id: {} for t in tracks}
    owners: dict[int, int | None] = {}
    smoking: list[tuple[int, Detection]] = []
    for index, detection in enumerate(detections):
        if detection.weak:
            # Diagnostic only: visible in the UI, never attached to a worker.
            owners[index] = None
            continue
        if detection.class_name in {"smoking"}:
            smoking.append((index, detection))
            owners[index] = None
            continue
        if detection.class_name not in {
            "helmet",
            "no_helmet",
            "vest",
            "no_vest",
            "mask",
            "no_mask",
        }:
            owners[index] = None
            continue
        box = detection.bbox
        center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        candidates = []
        for track in tracks:
            p = track.bbox
            w, h = p.x2 - p.x1, p.y2 - p.y1
            label = detection.class_name
            if label in {"helmet", "no_helmet"}:
                in_band = -0.15 <= (center[1] - p.y1) / max(h, 1e-6) <= 0.45
                target_y = 0.12
            elif label in {"mask", "no_mask"}:
                in_band = 0.0 <= (center[1] - p.y1) / max(h, 1e-6) <= 0.5
                target_y = 0.25
            else:  # vest torso band
                in_band = 0.15 <= (center[1] - p.y1) / max(h, 1e-6) <= 0.85
                target_y = 0.5
            relative_y = (center[1] - p.y1) / max(h, 1e-6)
            if (
                p.x1 - 0.1 * w <= center[0] <= p.x2 + 0.1 * w
                and in_band
                and _overlap_area(box, p) > 0
            ):
                distance = math.hypot(
                    (center[0] - (p.x1 + p.x2) / 2) / max(w, 1e-6),
                    relative_y - target_y,
                )
                candidates.append((distance, track.id))
        if candidates:
            _, key = min(candidates)
            owners[index] = key
            values = assignments[key]
            values[detection.class_name] = max(
                values.get(detection.class_name, 0), detection.confidence
            )
        else:
            owners[index] = None
    for index, detection in smoking:
        box = detection.bbox
        center = ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)
        best = None
        for track in tracks:
            p = track.bbox
            w, h = p.x2 - p.x1, p.y2 - p.y1
            face = ((p.x1 + p.x2) / 2, p.y1 + 0.3 * h)
            distance = math.hypot(
                (center[0] - face[0]) / max(w, 1e-6),
                (center[1] - face[1]) / max(h, 1e-6),
            )
            inside = (
                p.x1 - 0.3 * w <= center[0] <= p.x2 + 0.3 * w
                and p.y1 - 0.2 * h <= center[1] <= p.y2 + 0.2 * h
                and _overlap_area(box, p, margin=0.15) > 0
            )
            if inside and (best is None or distance < best[0]):
                best = (distance, track.id)
        if best is not None:
            owners[index] = best[1]
            values = assignments[best[1]]
            values["smoking"] = max(values.get("smoking", 0), detection.confidence)
    return assignments, owners


def evaluate_assigned(
    tracks: list[Track], assignments: dict[int, dict[str, float]], config: CameraConfig
):
    result = {}
    for track in tracks:
        values = assignments[track.id]
        for positive, negative, enabled in [
            ("helmet", "no_helmet", config.helmet),
            ("vest", "no_vest", config.vest),
            ("mask", "no_mask", config.mask),
        ]:
            pos, neg = values.get(positive, 0), values.get(negative, 0)
            track.ppe[positive] = "missing" if neg > pos else "present" if pos else "unknown"
            track.ppe_confidence[positive] = max(pos, neg)
            if enabled and neg > pos:
                result[(track.id, negative)] = neg
                track.violations.append(negative)
        smoking_conf = values.get("smoking", 0)
        if smoking_conf:
            track.ppe_confidence["smoking"] = smoking_conf
        if getattr(config, "smoking", False) and smoking_conf:
            result[(track.id, "smoking")] = smoking_conf
            track.violations.append("smoking")
        zones = config.effective_zones() if hasattr(config, "effective_zones") else []
        foot = ((track.bbox.x1 + track.bbox.x2) / 2, track.bbox.y2)
        for zone in zones:
            if contains(zone.points, foot):
                result[(track.id, "danger_zone")] = track.confidence
                track.violations.append("danger_zone")
                # Remember which named zone triggered, for the incident row.
                track.zone_name = zone.name
                break
    return result


def evaluate(tracks: list[Track], detections: list[Detection], config: CameraConfig):
    """Historic entry point: assignment + rules, observed violations only."""
    assignments, _ = assign_detections(tracks, detections, config)
    return evaluate_assigned(tracks, assignments, config)


def cleared_assigned(
    tracks: list[Track], assignments: dict[int, dict[str, float]], config: CameraConfig
) -> dict[tuple[int, str], float]:
    """Positive evidence that a violation is gone (the `present` side).

    - PPE pair: the positive class beats the negative one on the same worker.
    - smoking: no cigarette attached to the worker in this frame.
    - danger_zone: the worker's control point is outside the polygon.
    Absence of any evidence is `unknown` and never appears here.
    """
    cleared = {}
    for track in tracks:
        values = assignments[track.id]
        for positive, negative in [
            ("helmet", "no_helmet"),
            ("vest", "no_vest"),
            ("mask", "no_mask"),
        ]:
            if values.get(positive, 0) > values.get(negative, 0):
                cleared[(track.id, negative)] = values[positive]
        if not values.get("smoking", 0):
            cleared[(track.id, "smoking")] = track.confidence
        zones = config.effective_zones() if hasattr(config, "effective_zones") else []
        foot = ((track.bbox.x1 + track.bbox.x2) / 2, track.bbox.y2)
        if zones and not any(contains(zone.points, foot) for zone in zones):
            cleared[(track.id, "danger_zone")] = track.confidence
    return cleared


class ViolationPhase(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    CLEARING = "clearing"


@dataclass
class Violation:
    since: float
    last: float
    confidence: float
    incident_id: str | None = None
    phase: ViolationPhase = ViolationPhase.PENDING
    clear_since: float | None = None


class TemporalEngine:
    """PENDING → ACTIVE ↔ CLEARING → RESOLVED; observation gaps interrupt ACTIVE.

    Confirmation additionally requires temporal consensus: among the last
    `window` PROCESSED frames at least `required` must have observed the
    violation key. Every processed frame appends exactly one vote per known
    key — a miss is a recorded False vote even when the PENDING state is
    dropped, so the history is a literal rolling window, not an ambiguous
    counter. Histories are bounded deques; dead keys (no True vote left) are
    swept. Time (`confirm_seconds`) still applies on top of voting.
    """

    def __init__(self):
        self.states: dict[tuple[int, str], Violation] = {}
        self.votes: dict[tuple[int, str], deque] = {}
        self.previous: float | None = None

    def update(
        self, observed, now, confirm_seconds, clear_seconds,
        window=4, required=3, cleared=None,
    ):
        """Update with optional positive `cleared` evidence {key: confidence}.

        Semantics (sprint validation-reliability):
        - `missing` (in `observed`) confirms/continues the violation;
        - `present` (in `cleared`) is the ONLY path toward RESOLVED;
        - `unknown` (in neither) advances neither: a prolonged unknown emits
          "lost" instead of falsely claiming the violation cleared.
        """
        cleared = cleared or {}
        if self.previous is not None and now < self.previous:
            raise ValueError("Frame timestamps must be monotonic within a camera session.")
        needed = max(1, min(required, window))
        events = []
        gap = self.previous is not None and now - self.previous > 3
        self.previous = now
        if gap:
            self.votes.clear()
        # 1. Vote first: every known key gets exactly one vote per frame.
        for key in set(self.states) | set(observed) | set(self.votes):
            history = self.votes.get(key)
            if history is None or history.maxlen != window:
                history = deque(list(history or [])[-window:], maxlen=window)
                self.votes[key] = history
            history.append(key in observed)
        # 2. State transitions.
        for key in list(self.states):
            state = self.states[key]
            if gap:
                if state.incident_id:
                    events.append(("interrupted", key, state))
                del self.states[key]
            elif key not in observed:
                if state.phase == ViolationPhase.PENDING:
                    del self.states[key]
                else:
                    state.phase = ViolationPhase.CLEARING
                    if key in cleared:
                        if state.clear_since is None:
                            state.clear_since = now
                        if now - state.clear_since >= clear_seconds:
                            events.append(("resolved", key, state))
                            del self.states[key]
                            self.votes.pop(key, None)
                    else:
                        # Unknown: pause clearing, never resolve from ignorance.
                        state.clear_since = None
                        if now - state.last >= clear_seconds:
                            events.append(("lost", key, state))
                            del self.states[key]
                            self.votes.pop(key, None)
        for key, confidence in observed.items():
            state = self.states.setdefault(key, Violation(now, now, confidence))
            state.last = now
            state.confidence = max(state.confidence, confidence)
            if state.phase == ViolationPhase.CLEARING:
                state.phase = ViolationPhase.ACTIVE
                state.clear_since = None
            if (
                state.phase == ViolationPhase.PENDING
                and now - state.since >= confirm_seconds
                and sum(self.votes.get(key, ())) >= needed
            ):
                state.phase = ViolationPhase.ACTIVE
                events.append(("active", key, state))
        for key in list(self.votes):
            if (
                key not in self.states
                and key not in observed
                and not any(self.votes[key])
            ):
                del self.votes[key]
        return events

    def describe(self, now, confirm_seconds):
        return [
            {
                "track_id": key[0],
                "type": key[1],
                "phase": state.phase,
                "confidence": state.confidence,
                "elapsed_seconds": round(state.last - state.since, 2),
                "confirm_seconds": confirm_seconds,
                "incident_id": state.incident_id,
            }
            for key, state in self.states.items()
        ]
