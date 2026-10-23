from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, model_validator

from app.domain.models import BoundingBox, Detection
from app.services.state_machine import CameraState


class Zone(BaseModel):
    """A named danger zone. Points are normalized [0, 1]."""
    id: str = Field(default="", max_length=32)
    name: str = Field(default="Опасная зона", min_length=1, max_length=80)
    enabled: bool = True
    points: list[tuple[float, float]] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def valid_points(self):
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.points):
            raise ValueError("Zone coordinates must be in 0..1.")
        if self.enabled and len(self.points) < 3:
            raise ValueError("An enabled zone needs at least three points.")
        if self.points:
            CameraConfig(zone=self.points)
        return self


class CameraConfig(BaseModel):
    name: str = Field(default="Камера 1", min_length=1, max_length=80)
    # Owner scope: None = shared operator camera; "phone:<id>" = isolated
    # to one phone pairing. Set only by the pairing flow, never by the form.
    owner: str | None = None
    helmet: bool = True
    vest: bool = True
    # Post-MVP extensions (PDF §11: MVP scope is helmet/vest/zone).
    # Off by default; enable per camera once the model provides the classes.
    mask: bool = False
    smoking: bool = False
    # Legacy single zone. Kept for old clients and stored cameras.
    zone_enabled: bool = False
    zone_name: str = Field(default="Опасная зона", min_length=1, max_length=80)
    zone: list[tuple[float, float]] = Field(default_factory=list, max_length=30)
    # Named zones. When present they replace the legacy single zone for
    # monitoring; the legacy fields mirror the first enabled zone so older
    # clients and incident filters keep working.
    zones: list[Zone] = Field(default_factory=list, max_length=10)
    # Behavioural analytics (case slide 4, scenario 01). Runs on Worker
    # tracks only: no model class is required, so it works with any detector.
    running_enabled: bool = False
    running_speed: float = Field(default=1.8, ge=0.5, le=10)
    running_min_seconds: float = Field(default=0.8, ge=0.2, le=10)
    confirm_seconds: float = Field(default=2, ge=0.2, le=60)
    clear_seconds: float = Field(default=2, ge=0.2, le=60)
    # Temporal consensus (§4): confirm only after `temporal_required`
    # positive observations within the last `temporal_window` frames.
    temporal_window: int = Field(default=4, ge=1, le=20)
    temporal_required: int = Field(default=3, ge=1, le=20)

    def effective_zones(self) -> list[Zone]:
        """Enabled, drawable zones used by the radar and the overlay."""
        if self.zones:
            return [z for z in self.zones if z.enabled and len(z.points) >= 3]
        if self.zone_enabled and len(self.zone) >= 3:
            return [Zone(id="legacy", name=self.zone_name, enabled=True, points=self.zone)]
        return []

    @model_validator(mode="after")
    def valid_temporal(self):
        if self.temporal_required > self.temporal_window:
            raise ValueError("temporal_required must not exceed temporal_window.")
        return self.valid_zone()

    @model_validator(mode="after")
    def valid_zone(self):
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.zone):
            raise ValueError("Zone coordinates must be in 0..1.")
        if self.zone and len(self.zone) < 3:
            raise ValueError("A zone needs at least three points.")
        if self.zone:

            def cross(a, b, c):
                return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

            for i, a in enumerate(self.zone):
                b = self.zone[(i + 1) % len(self.zone)]
                if a == b:
                    raise ValueError("Zone must not contain duplicate adjacent points.")
                for j in range(i + 1, len(self.zone)):
                    if j == i + 1 or (i == 0 and j == len(self.zone) - 1):
                        continue
                    c, d = self.zone[j], self.zone[(j + 1) % len(self.zone)]
                    if (
                        cross(a, b, c) * cross(a, b, d) <= 0
                        and cross(c, d, a) * cross(c, d, b) <= 0
                    ):
                        if max(min(a[0], b[0]), min(c[0], d[0])) <= min(
                            max(a[0], b[0]), max(c[0], d[0])
                        ) and max(min(a[1], b[1]), min(c[1], d[1])) <= min(
                            max(a[1], b[1]), max(c[1], d[1])
                        ):
                            raise ValueError("Zone edges must not intersect.")
            area = (
                abs(
                    sum(
                        x * self.zone[(i + 1) % len(self.zone)][1]
                        - self.zone[(i + 1) % len(self.zone)][0] * y
                        for i, (x, y) in enumerate(self.zone)
                    )
                )
                / 2
            )
            if area < 0.0001:
                raise ValueError("Zone area must be greater than zero.")
        if len({z.name for z in self.zones}) != len(self.zones):
            raise ValueError("Zone names must be unique.")
        if self.zones:
            enabled = [z for z in self.zones if z.enabled]
            if enabled:
                self.zone = list(enabled[0].points)
                self.zone_name = enabled[0].name
                self.zone_enabled = True
            else:
                self.zone_enabled = False
            return self
        if self.zone_enabled and not self.zone:
            raise ValueError("Draw a zone before enabling it.")
        return self


class StartSource(BaseModel):
    kind: Literal["browser", "rtsp"] = "browser"
    url: str = Field(default="", max_length=2048)

    @model_validator(mode="after")
    def valid_source(self):
        if self.kind == "rtsp":
            if not self.url.strip():
                raise ValueError("Укажите адрес видеопотока.")
            scheme = urlsplit(self.url.strip().lower()).scheme
            # SSRF guard: only stream schemes reach OpenCV/FFmpeg. Credentials
            # inside rtsp(s) URLs stay allowed (cameras need them); probing
            # internal networks is a documented trusted-operator risk.
            if scheme not in {"rtsp", "rtsps", "http", "https"}:
                raise ValueError("RTSP-источник должен начинаться с rtsp://, rtsps:// или http(s)://.")
        return self


class Feedback(BaseModel):
    verdict: Literal["confirmed", "false_positive", "unreviewed"]


class PairingRequest(BaseModel):
    camera_id: str | None = Field(default=None, min_length=1, max_length=64)
    # Alternatively mint a fresh isolated camera owned by the phone token.
    new_camera_name: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def exactly_one_target(self):
        if bool(self.camera_id) == bool(self.new_camera_name):
            raise ValueError("Укажите camera_id или new_camera_name (ровно одно).")
        return self


class PairingExchange(BaseModel):
    code: str = Field(min_length=1, max_length=64)


# Public HTTP contract: domain output is validated before being sent to the frontend.


class TrackView(BaseModel):
    id: int
    bbox: BoundingBox
    confidence: float = Field(ge=0, le=1)
    violations: list[str]
    ppe: dict[str, Literal["present", "missing", "unknown"]]
    ppe_confidence: dict[str, float]
    ppe_age_seconds: float = 0.0


class RuleView(BaseModel):
    track_id: int
    type: Literal[
        "no_helmet", "no_vest", "no_mask", "smoking", "danger_zone", "running"
    ]
    phase: Literal["pending", "active", "clearing"]
    confidence: float = Field(ge=0, le=1)
    elapsed_seconds: float = Field(ge=0)
    confirm_seconds: float = Field(gt=0)
    incident_id: str | None


class RadarEventView(BaseModel):
    track_id: int
    camera_id: str
    risk_type: Literal["zone_approach", "zone_cross", "worker_cross"]
    level: Literal["WARNING", "CRITICAL", "UNKNOWN"]
    timestamp: float
    message: str
    zone_name: str | None = None
    other_track_id: int | None = None
    evidence_url: str | None = None

class RadarTrackView(BaseModel):
    track_id: int
    level: Literal["NORMAL", "WARNING", "CRITICAL", "UNKNOWN"]
    trajectory: list[tuple[float, float]]
    predicted: list[tuple[float, float]]
    direction: tuple[float, float] | None = None
    intersection: tuple[float, float] | None = None
    confidence: float = 0.0

class RadarZoneView(BaseModel):
    name: str
    points: list[tuple[float, float]]

class RadarSceneView(BaseModel):
    tracks: list[RadarTrackView]
    zones: list[RadarZoneView]

class RadarView(BaseModel):
    track_risks: dict[int, Literal["NORMAL", "WARNING", "CRITICAL", "UNKNOWN"]]
    events: list[RadarEventView]
    scene: RadarSceneView | None = None

class FrameResult(BaseModel):
    session_id: str
    sequence: int = Field(ge=1)
    frame_url: str
    tracks: list[TrackView]
    detections: list[Detection]
    rules: list[RuleView]
    latency_ms: float = Field(ge=0)
    fps: float = Field(ge=0)
    timestamp: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    active: int = Field(ge=0)
    radar: RadarView | None = None
    metrics: dict | None = None
    source_timestamp: float | None = None
    data_age_ms: float = 0.0


class CameraView(CameraConfig):
    frame_capacity: int = 1
    id: str
    status: CameraState
    error: str | None
    kind: Literal["browser", "video", "rtsp"] | None
    latest: FrameResult | None
    session_id: str | None
    state_version: int = Field(ge=0)
