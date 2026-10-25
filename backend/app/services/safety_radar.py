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
from uuid import uuid4

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
    episode_id: str | None = None
    lead_time_seconds: float | None = None
    predicted_entry_seconds: float | None = None


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
        self._zone_state = {}
        self._warning_since = {}
        self._episode_ids = {}
        self._outside_streak = {}
        self._episode_inside = {}

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
        for track_id in self._state:
            self._state[track_id] = UNKNOWN
        self._trajectories.clear()
        self._direction.clear()
        self._speed.clear()
        self._zone_state.clear()
        self._warning_since.clear()
        self._episode_ids.clear()
        self._outside_streak.clear()
        self._episode_inside.clear()
        self._events.clear()
        self._last_notified.clear()

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
        self._zone_state.clear()
        self._warning_since.clear()
        self._episode_ids.clear()
        self._outside_streak.clear()
        self._episode_inside.clear()

    # -- per-frame update ------------------------------------------------

    def update(self, tracks: list[Track], camera_id: str, frame_time: float) -> list[RiskEvent]:
        active, new_events = set(), []
        for track in tracks:
            active.add(track.id)
            previous_seen = self._last_seen.get(track.id)
            if previous_seen is not None and frame_time <= previous_seen:
                continue
            if previous_seen is not None and frame_time - previous_seen > self.lost_grace_seconds:
                self._trajectories.pop(track.id, None)
                for key in list(self._zone_state):
                    if key[0] == track.id:
                        self._zone_state[key] = UNKNOWN
                        self._warning_since.pop(key, None)
            self._last_seen[track.id] = frame_time
            x, y = self._foot_point(track)
            traj = self._trajectories.setdefault(track.id, deque())
            traj.append(TrajectoryPoint(x, y, frame_time, track.id))
            while traj and traj[0].t < frame_time - self.history_seconds:
                traj.popleft()
            prediction = self.predict(track.id, frame_time)
            point = Point(x, y)
            levels = []
            for name, polygon in self._zones.items():
                key = (track.id, name)
                previous = self._zone_state.get(key, UNKNOWN)
                if polygon.covers(point):
                    self._outside_streak[key] = 0
                    level, kind = CRITICAL, "zone_cross"
                elif previous == CRITICAL and (
                    point.distance(polygon) < 0.005 or self._outside_streak.get(key, 0) < 1
                ):
                    self._outside_streak[key] = self._outside_streak.get(key, 0) + 1
                    level, kind = CRITICAL, "zone_cross"
                elif len(traj) < self.min_history_points:
                    level, kind = UNKNOWN, ""
                elif (
                    prediction
                    and prediction.get("zone_name") == name
                    and prediction["confidence"] >= 0.5
                ):
                    level, kind = WARNING, "zone_approach"
                else:
                    level, kind = NORMAL, ""
                levels.append(level)
                self._zone_state[key] = level
                confirmed_exit = self._episode_inside.get(key, False) and level in {NORMAL, WARNING}
                if level == CRITICAL:
                    self._episode_inside[key] = True
                elif confirmed_exit:
                    self._episode_inside[key] = False
                if level not in {WARNING, CRITICAL}:
                    self._warning_since.pop(key, None)
                    continue
                if level == WARNING and previous != WARNING:
                    self._warning_since[key] = frame_time
                if previous == NORMAL or confirmed_exit or key not in self._episode_ids:
                    self._episode_ids[key] = uuid4().hex
                # State transitions, not every notification interval, define episodes.
                # An actual re-entry bypasses the old notification cooldown.
                if level == previous:
                    continue
                warned = self._warning_since.get(key)
                lead = (
                    frame_time - warned
                    if level == CRITICAL and previous == WARNING and warned is not None
                    else None
                )
                distance = float(point.distance(polygon))
                event = RiskEvent(
                    track.id,
                    camera_id,
                    kind,
                    level,
                    frame_time,
                    self._message(track.id, level, kind, name, distance),
                    trajectory=[(p.x, p.y) for p in traj],
                    zone_name=name,
                    episode_id=self._episode_ids.get(key),
                    lead_time_seconds=round(lead, 4) if lead is not None else None,
                    predicted_entry_seconds=prediction.get("time_to_boundary_seconds")
                    if prediction and level == WARNING
                    else None,
                )
                self._events.append(event)
                new_events.append(event)
                if len(self._events) > self.max_events:
                    del self._events[: len(self._events) - self.max_events]
            self._state[track.id] = next(
                (v for v in (CRITICAL, WARNING, UNKNOWN) if v in levels),
                NORMAL if levels else UNKNOWN,
            )
        for tid, seen in list(self._last_seen.items()):
            if tid not in active and frame_time - seen > self.lost_grace_seconds:
                self._state[tid] = UNKNOWN
                for key in list(self._zone_state):
                    if key[0] == tid:
                        self._zone_state[key] = UNKNOWN
                        self._warning_since.pop(key, None)
                if frame_time - seen > self.history_seconds:
                    self._trajectories.pop(tid, None)
                    self._last_seen.pop(tid, None)
                    self._state.pop(tid, None)
                    for mapping in (
                        self._zone_state,
                        self._episode_ids,
                        self._outside_streak,
                        self._episode_inside,
                    ):
                        for key in list(mapping):
                            if key[0] == tid:
                                mapping.pop(key, None)
        return new_events

    # -- geometry --------------------------------------------------------

    @staticmethod
    def _foot_point(track: Track) -> tuple[float, float]:
        box = track.bbox
        return (box.x1 + box.x2) / 2.0, box.y2

    def _classify(self, traj: deque[TrajectoryPoint]) -> tuple[str, str, str | None, float]:
        if not self._zones or not traj:
            return UNKNOWN, "", None, 0.0
        point = Point(traj[-1].x, traj[-1].y)
        for name, polygon in self._zones.items():
            if polygon.covers(point):
                return CRITICAL, "zone_cross", name, 0.0
        prediction = self.predict(traj[-1].track_id, traj[-1].t)
        if prediction and prediction.get("zone_name"):
            name = prediction["zone_name"]
            return WARNING, "zone_approach", name, float(point.distance(self._zones[name]))
        return (NORMAL if len(traj) >= self.min_history_points else UNKNOWN), "", None, 0.0

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

    @staticmethod
    def _entry(line, polygon):
        if polygon.covers(Point(line.coords[0])) or not line.crosses(polygon):
            return None
        hit = line.intersection(polygon.boundary)
        geometries = list(hit.geoms) if hasattr(hit, "geoms") else [hit]
        candidates = []
        for geometry in geometries:
            points = (
                [geometry]
                if geometry.geom_type == "Point"
                else [Point(c) for c in geometry.coords]
                if hasattr(geometry, "coords")
                else []
            )
            for point in points:
                distance = line.project(point)
                after = line.interpolate(min(line.length, distance + 1e-6))
                if distance > 1e-9 and polygon.contains(after):
                    candidates.append((distance, (float(point.x), float(point.y))))
        return min(candidates, default=None)

    def predict(
        self,
        track_id: int,
        frame_time: float,
        horizon_seconds: float = 1.5,
        min_speed: float = 0.01,
        max_gap_seconds: float = 1.5,
    ) -> dict | None:
        traj = self._trajectories.get(track_id)
        if not traj or len(traj) < self.min_history_points:
            return None
        last = traj[-1]
        if frame_time - last.t > max_gap_seconds:
            return None
        window = [p for p in traj if last.t - p.t <= 1.0]
        if len(window) < self.min_history_points:
            window = list(traj)[-self.min_history_points :]
        times = np.array([p.t for p in window], dtype=float)
        gaps = np.diff(times)
        if not len(gaps) or np.any(gaps <= 0) or max(gaps) > max_gap_seconds:
            return None
        points = np.array([(p.x, p.y) for p in window])
        # Stop/reversal gate uses the latest real observation, so old smoothed
        # velocity cannot continue predicting an entry after stopping.
        recent = float(np.linalg.norm(points[-1] - points[-2]) / gaps[-1])
        if recent < min_speed:
            return None
        centered = times - times.mean()
        denom = float(centered @ centered)
        if denom <= 1e-10:
            return None
        velocity = (centered[:, None] * (points - points.mean(axis=0))).sum(axis=0) / denom
        speed = float(np.linalg.norm(velocity))
        path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        net = float(np.linalg.norm(points[-1] - points[0]))
        straightness = net / path if path > 1e-9 else 0.0
        if speed < min_speed or straightness < 0.65:
            return None
        fit = points.mean(axis=0) + centered[:, None] * velocity
        residual = float(np.linalg.norm(points - fit, axis=1).mean())
        confidence = round(max(0.0, min(1.0, straightness * (1 - residual / max(net, 0.001)))), 3)
        end = points[-1] + velocity * horizon_seconds
        line = LineString([tuple(points[-1]), tuple(end)])
        entries = []
        for name, polygon in self._zones.items():
            entry = self._entry(line, polygon)
            if entry:
                entries.append((entry[0], name, entry[1]))
        nearest = min(entries, default=None)
        return {
            "track_id": track_id,
            "trajectory": [(p.x, p.y) for p in traj][-20:],
            "predicted": [tuple(points[-1]), tuple(end)],
            "direction": tuple(velocity / speed),
            "speed_norm": round(speed, 4),
            "intersection": nearest[2] if nearest else None,
            "zone_name": nearest[1] if nearest else None,
            "time_to_boundary_seconds": round(nearest[0] / speed, 4) if nearest else None,
            "confidence": confidence,
        }

    def snapshot(self, frame_time: float) -> dict:
        """Per-track state for the UI: level, trajectory, prediction."""
        tracks = []
        for track_id in list(self._trajectories):
            prediction = self.predict(track_id, frame_time)
            tracks.append(
                {
                    "track_id": track_id,
                    "level": self.get_current_risk_level(track_id, frame_time),
                    "trajectory": (
                        prediction["trajectory"]
                        if prediction
                        else [(p.x, p.y) for p in self._trajectories[track_id]][-20:]
                    ),
                    "predicted": prediction["predicted"] if prediction else [],
                    "direction": prediction["direction"] if prediction else None,
                    "intersection": prediction["intersection"] if prediction else None,
                    "confidence": prediction["confidence"] if prediction else 0.0,
                    "predicted_zone": prediction["zone_name"] if prediction else None,
                    "time_to_boundary_seconds": prediction["time_to_boundary_seconds"]
                    if prediction
                    else None,
                }
            )
        zones = [
            {"name": name, "points": list(polygon.exterior.coords)[:-1]}
            for name, polygon in self._zones.items()
        ]
        return {"tracks": tracks, "zones": zones}

    # -- read side -------------------------------------------------------

    def get_current_risk_level(self, track_id: int, frame_time: float | None = None) -> str:
        """Continuous state; cooldown never downgrades it."""
        if (
            frame_time is not None
            and frame_time - self._last_seen.get(track_id, frame_time) > self.lost_grace_seconds
        ):
            return UNKNOWN
        return self._state.get(track_id, UNKNOWN)

    def get_events(self, since: float = 0) -> list[RiskEvent]:
        """Latest event per (track, risk_type) for the UI; full history stays internal."""
        latest: dict[tuple[int, str], RiskEvent] = {}
        for event in self._events:
            if event.timestamp >= since:
                latest[(event.track_id, event.risk_type, event.zone_name)] = event
        return sorted(latest.values(), key=lambda e: e.timestamp)

    def zone_level(self, track_id, zone_name):
        return self._zone_state.get((track_id, zone_name), UNKNOWN)

    def episode_id(self, track_id, zone_name):
        return self._episode_ids.get((track_id, zone_name))

    def current_risks(self, frame_time):
        return {
            tid: self.get_current_risk_level(tid, frame_time)
            for tid, seen in self._last_seen.items()
            if frame_time - seen <= self.history_seconds
        }

    def all_events(self) -> list[RiskEvent]:
        return list(self._events)
