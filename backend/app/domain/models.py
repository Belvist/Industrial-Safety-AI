from dataclasses import dataclass
from enum import StrEnum
from math import isfinite


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        if not all(isfinite(v) and 0 <= v <= 1 for v in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("Bounding boxes must use finite normalized 0..1 coordinates.")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("Bounding box must have positive width and height.")


@dataclass(frozen=True, slots=True)
class Detection:
    """Backend-neutral detector output used by all downstream modules."""

    class_name: str
    confidence: float
    bbox: BoundingBox
    track_hint: str | None = None
    # Weak diagnostic detection: below the per-class violation threshold but
    # above the display floor. Returned for visibility, never creates or
    # clears violations (engine assignment skips weak).
    weak: bool = False
    # Raw pre-smoothing confidence. Set only on display-smoothed copies so the
    # shown confidence/weak pair stays self-consistent while the raw evidence
    # remains auditable. Engine and matching always use `confidence` of the
    # RAW objects, never display copies.
    raw_confidence: float | None = None

    def __post_init__(self):
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be in 0..1.")
        if self.raw_confidence is not None and (
            not isfinite(self.raw_confidence) or not 0 <= self.raw_confidence <= 1
        ):
            raise ValueError("Detection raw_confidence must be in 0..1.")
        if not self.class_name:
            raise ValueError("Detection class_name must not be empty.")


class IncidentStatus(StrEnum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    INTERRUPTED = "interrupted"


class IncidentType(StrEnum):
    NO_HELMET = "no_helmet"
    NO_VEST = "no_vest"
    NO_MASK = "no_mask"
    SMOKING = "smoking"
    DANGER_ZONE = "danger_zone"


# Canonical detector labels consumed by tracking/rules.
# LinQ and local ONNX adapters must normalize to these names.
PERSON_LABEL = "person"
HELMET_LABELS = {"helmet", "no_helmet"}
VEST_LABELS = {"vest", "no_vest"}
MASK_LABELS = {"mask", "respirator", "no_mask"}
SMOKING_LABELS = {"cigarette", "smoking", "smoke"}

VIOLATION_LABELS = {
    IncidentType.NO_HELMET,
    IncidentType.NO_VEST,
    IncidentType.NO_MASK,
    IncidentType.SMOKING,
    IncidentType.DANGER_ZONE,
}
