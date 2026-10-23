"""Safety Radar: trajectory analysis and early warning.

Separate from the inference adapter. It consumes confirmed person Tracks
from the tracker (works identically for LinQ and local ONNX) and keeps:

- continuous risk state per track (NORMAL/WARNING/CRITICAL/UNKNOWN);
- notification throttling (cooldown), which never clears the state;
- explainable events with trajectory for Evidence Replay.

Coordinates are normalized [0, 1] end to end: zones, foot points, distances
and trajectories. No metres, m/s or time-to-collision are produced without
camera calibration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from app.services.engine import Track

NORMAL = "NORMAL"
WARNING = "WARNING"
CRITICAL = "CRITICAL"
UNKNOWN = "UNKNOWN"


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    t: float
    track_id: int


@dataclass
class RiskEvent:
    track_id: int
    camera_id: str
    risk_type: str  # zone_approach | zone_cross
    level: str  # WARNING | CRITICAL
    timestamp: float
    message: str
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    zone_name: str | None = None
    other_track_id: int | None = None
    evidence: str | None = None


class SafetyRadar:
    """Continuous risk state plus throttled, explainable notifications."""

    def __init__(
        self,
        history_seconds: float = 5.0,
        zone_threshold_norm: float = 0.02,
        speed_threshold_norm_s: float = 0.5,
        min_history_points: int = 3,
        cooldown_seconds: float = 3.0,
        lost_grace_seconds: float = 1.0,
        max_events: int = 1000,
    ) -> None:
        self.history_seconds = history_seconds
        self.zone_threshold_norm = zone_threshold_norm
        self.speed_threshold_norm_s = speed_threshold_norm_s
        self.min_history_points = min_history_points
        self.cooldown_seconds = cooldown_seconds
        self.lost_grace_seconds = lost_grace_seconds
        self.max_events = max_events

        self._trajectories: dict[int, deque[TrajectoryPoint]] = {}
        # Continuous state, independent of notification cooldown.
        self._state: dict[int, str] = {}
        self._last_seen: dict[int, float] = {}
        # consecutive NORMAL frames required to leave WARNING (hysteresis)
        self._safe_streak: dict[int, int] = {}
        self._previous_type: dict[int, tuple[str, str | None, float]] = {}
        self.clear_frames = 3
        # Throttling only: (track_id, risk_type) -> last notification time.
        self._last_notified: dict[tuple[int, str], float] = {}
        self._events: list[RiskEvent] = []
        self._zones: dict[str, Polygon] = {}
        # Smoothed direction per track (unit vector) for cautious prediction.
        self._direction: dict[int, tuple[float, float]] = {}
        self._speed: dict[int, float] = {}

    # -- configuration ---------------------------------------------------

    def set_zones(self, zones: dict[str, list[tuple[float, float]]], enabled: bool = True) -> None:
        """Replace zones. Disabled/empty zones invalidate prior risk state."""
        if not enabled or not zones:
            self._zones = {}
            self._invalidate_state()
            # A disabled zone must not keep showing old alarms in the UI.
            self._events.clear()
            self._last_notified.clear()
            return
        new = {name: Polygon(points) for name, points in zones.items() if len(points) >= 3}
        if new.keys() != self._zones.keys() or any(
            not new[name].equals(self._zones[name]) for name in new
        ):
            self._invalidate_state()
        self._zones = new

    def _invalidate_state(self) -> None:
        # Geometry changed: old conclusions are not valid evidence.
        for track_id in list(self._state):
            self._state[track_id] = UNKNOWN

    def reset(self) -> None:
        self._trajectories.clear()
        self._state.clear()
        self._last_seen.clear()
        self._last_notified.clear()
        self._safe_streak.clear()
        self._previous_type.clear()
        self._events.clear()
        self._zones.clear()
        self._direction.clear()
        self._speed.clear()

    # -- per-frame update ------------------------------------------------

    def update(self, tracks: list[Track], camera_id: str, frame_time: float) -> list[RiskEvent]:
        active: set[int] = set()
        new_events: list[RiskEvent] = []

        for track in tracks:
            active.add(track.id)
            self._last_seen[track.id] = frame_time
            x, y = self._foot_point(track)

            traj = self._trajectories.setdefault(track.id, deque())
            traj.append(TrajectoryPoint(x, y, frame_time, track.id))
            cutoff = frame_time - self.history_seconds
            while traj and traj[0].t < cutoff:
                traj.popleft()

            level, risk_type, zone_name, dist = self._classify(traj)
            previous = self._state.get(track.id, UNKNOWN)
            if level == NORMAL:
                self._safe_streak[track.id] = self._safe_streak.get(track.id, 0) + 1
                # Keep WARNING through brief direction jitter.
                if previous == WARNING and self._safe_streak[track.id] < self.clear_frames:
                    level = WARNING
                    risk_type, zone_name, dist = self._previous_type.get(
                        track.id, ("zone_approach", None, 0.0)
                    )
            else:
                self._safe_streak[track.id] = 0
            self._state[track.id] = level

            # UNKNOWN is a state, not a notifiable event.
            if level == WARNING:
                self._previous_type[track.id] = (risk_type, zone_name, dist)
            if level not in (WARNING, CRITICAL):
                continue
            # Notify on first detection, on escalation, or after cooldown.
            key = (track.id, risk_type)
            last = self._last_notified.get(key)
            escalated = level == CRITICAL and previous != CRITICAL
            if last is None or escalated or frame_time - last >= self.cooldown_seconds:
                self._last_notified[key] = frame_time
                event = RiskEvent(
                    track_id=track.id,
                    camera_id=camera_id,
                    risk_type=risk_type,
                    level=level,
                    timestamp=frame_time,
                    message=self._message(track.id, level, risk_type, zone_name, dist),
                    trajectory=[(p.x, p.y) for p in traj],
                    zone_name=zone_name,
                )
                self._events.append(event)
                new_events.append(event)
                if len(self._events) > self.max_events:
                    del self._events[: len(self._events) - self.max_events]

        # Lost observations become UNKNOWN after a short grace period; the
        # state is never silently reported as NORMAL.
        for track_id, seen in list(self._last_seen.items()):
            if track_id in active:
                continue
            if frame_time - seen > self.lost_grace_seconds:
                self._state[track_id] = UNKNOWN
                traj = self._trajectories.get(track_id)
                if traj is not None and (not traj or frame_time - traj[-1].t > self.history_seconds):
                    del self._trajectories[track_id]
                    self._last_seen.pop(track_id, None)

        return new_events

    # -- geometry --------------------------------------------------------

    @staticmethod
    def _foot_point(track: Track) -> tuple[float, float]:
        box = track.bbox
        return (box.x1 + box.x2) / 2.0, box.y2

    def _classify(self, traj: deque[TrajectoryPoint]) -> tuple[str, str, str | None, float]:
        if not self._zones:
            # No active zone geometry: safety cannot be confirmed.
            return UNKNOWN, "", None, 0.0
        if len(traj) < self.min_history_points:
            return UNKNOWN, "", None, 0.0

        point = Point(traj[-1].x, traj[-1].y)
        best_warning: tuple[str, float] | None = None
        for name, polygon in self._zones.items():
            if polygon.contains(point) or polygon.touches(point):
                return CRITICAL, "zone_cross", name, 0.0
            distance = point.distance(polygon)
            if distance <= self.zone_threshold_norm and self._moving_towards(traj, polygon):
                if best_warning is None or distance < best_warning[1]:
                    best_warning = (name, distance)
        if best_warning:
            return WARNING, "zone_approach", best_warning[0], best_warning[1]
        return NORMAL, "", None, 0.0

    @staticmethod
    def _moving_towards(traj: deque[TrajectoryPoint], zone: Polygon) -> bool:
        if len(traj) < 2:
            return False
        p1, p2 = traj[-2], traj[-1]
        vx, vy = p2.x - p1.x, p2.y - p1.y
        if vx == 0 and vy == 0:
            return False
        center = zone.centroid
        ux, uy = center.x - p2.x, center.y - p2.y
        norm_v = float(np.hypot(vx, vy))
        norm_u = float(np.hypot(ux, uy))
        if norm_v == 0 or norm_u == 0:
            return False
        return (vx * ux + vy * uy) / (norm_v * norm_u) > 0.3

    def _message(
        self,
        track_id: int,
        level: str,
        risk_type: str,
        zone_name: str | None,
        distance: float,
    ) -> str:
        if risk_type == "zone_cross":
            return f"Работник #{track_id} вошёл в зону «{zone_name}»"
        return (
            f"Работник #{track_id} приближается к зоне «{zone_name}» "
            f"(дистанция {distance:.3f} норм.)"
        )

    # -- short-horizon prediction ---------------------------------------

    def predict(
        self,
        track_id: int,
        frame_time: float,
        horizon_seconds: float = 1.5,
        min_speed: float = 0.01,
        max_gap_seconds: float = 1.5,
    ) -> dict | None:
        """Cautious constant-velocity projection of the foot point.

        Returns None when the observation is too sparse, too slow, or stale.
        This is normalized-image geometry only: no metres, m/s or TTC.
        """
        traj = self._trajectories.get(track_id)
        if traj is None or len(traj) < self.min_history_points:
            return None
        last = traj[-1]
        if frame_time - last.t > max_gap_seconds:
            return None

        # Use a short window to smooth jitter (up to ~1 s).
        window = [p for p in traj if last.t - p.t <= 1.0] or list(traj)
        first = window[0]
        dt = last.t - first.t
        if dt <= 0:
            return None
        vx = (last.x - first.x) / dt
        vy = (last.y - first.y) / dt
        speed = float(np.hypot(vx, vy))
        if speed < min_speed:
            return None

        # Exponential smoothing against the previous direction.
        ux, uy = vx / speed, vy / speed
        prev = self._direction.get(track_id)
        if prev is not None:
            ux = 0.6 * ux + 0.4 * prev[0]
            uy = 0.6 * uy + 0.4 * prev[1]
            norm = float(np.hypot(ux, uy)) or 1.0
            ux, uy = ux / norm, uy / norm
        self._direction[track_id] = (ux, uy)
        self._speed[track_id] = speed

        end = (last.x + ux * speed * horizon_seconds,
               last.y + uy * speed * horizon_seconds)
        predicted = [(last.x, last.y), end]

        intersection = None
        for polygon in self._zones.values():
            line = LineString(predicted)
            if line.intersects(polygon.boundary):
                hit = line.intersection(polygon.boundary)
                if hit.is_empty:
                    continue
                pt = hit.geoms[0] if hasattr(hit, "geoms") else hit
                intersection = (float(pt.x), float(pt.y))
                break

        # Reliability falls with sparse history and long gaps.
        freshness = max(0.0, 1.0 - (frame_time - last.t) / max_gap_seconds)
        density = min(1.0, len(window) / max(1, self.min_history_points * 4))
        confidence = round(min(1.0, 0.5 * freshness + 0.5 * density), 2)

        return {
            "track_id": track_id,
            "trajectory": [(p.x, p.y) for p in traj][-20:],
            "predicted": predicted,
            "direction": (ux, uy),
            "speed_norm": round(speed, 4),
            "intersection": intersection,
            "confidence": confidence,
        }

    def snapshot(self, frame_time: float) -> dict:
        """Per-track state for the UI: level, trajectory, prediction."""
        tracks = []
        for track_id in list(self._trajectories):
            prediction = self.predict(track_id, frame_time)
            tracks.append({
                "track_id": track_id,
                "level": self.get_current_risk_level(track_id),
                "trajectory": (
                    prediction["trajectory"] if prediction
                    else [(p.x, p.y) for p in self._trajectories[track_id]][-20:]
                ),
                "predicted": prediction["predicted"] if prediction else [],
                "direction": prediction["direction"] if prediction else None,
                "intersection": prediction["intersection"] if prediction else None,
                "confidence": prediction["confidence"] if prediction else 0.0,
            })
        zones = [
            {"name": name, "points": list(polygon.exterior.coords)[:-1]}
            for name, polygon in self._zones.items()
        ]
        return {"tracks": tracks, "zones": zones}

    # -- read side -------------------------------------------------------

    def get_current_risk_level(self, track_id: int, frame_time: float | None = None) -> str:
        """Continuous state; cooldown never downgrades it."""
        return self._state.get(track_id, UNKNOWN)

    def get_events(self, since: float = 0) -> list[RiskEvent]:
        """Latest event per (track, risk_type) for the UI; full history stays internal."""
        latest: dict[tuple[int, str], RiskEvent] = {}
        for event in self._events:
            if event.timestamp >= since:
                latest[(event.track_id, event.risk_type)] = event
        return sorted(latest.values(), key=lambda e: e.timestamp)

    def all_events(self) -> list[RiskEvent]:
        return list(self._events)
