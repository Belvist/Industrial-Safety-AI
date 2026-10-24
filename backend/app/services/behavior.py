"""Behavioural analytics over tracked workers.

Scenario 01 of the LinQ case (running / sharp movement) is derived from the
trajectory that the tracker already produces. It needs no extra model class,
so it works with every inference provider and keeps the hardware boundary
untouched.

Speed is normalised by the worker's own box height (person-heights per
second), which makes the threshold independent of resolution and camera
distance. Two guards keep a shaky camera or a jittering box from firing:
a minimum observation span and a straightness ratio (net displacement over
travelled path).
"""

import math
from collections import deque
from dataclasses import dataclass, field
from itertools import islice

from app.services.engine import Track


@dataclass
class MotionSample:
    time: float
    x: float
    y: float
    height: float


@dataclass
class _TrackMotion:
    samples: deque[MotionSample] = field(default_factory=deque)
    fast_since: float | None = None
    last_speed: float = 0.0


class RunningDetector:
    """Temporal trajectory rule for scenario 01 (running).

    ``update`` returns the same ``(observed, cleared)`` contract that the PPE
    evaluator uses, so the existing TemporalEngine handles confirmation,
    deduplication, RESOLVED and INTERRUPTED without special cases.
    """

    def __init__(
        self,
        history_seconds: float = 1.5,
        min_span_seconds: float = 0.4,
        min_samples: int = 3,
        straightness: float = 0.55,
        min_height: float = 0.05,
    ) -> None:
        self.history_seconds = history_seconds
        self.min_span_seconds = min_span_seconds
        self.min_samples = min_samples
        self.straightness = straightness
        self.min_height = min_height
        self._motions: dict[int, _TrackMotion] = {}

    def reset(self) -> None:
        self._motions.clear()

    def update(
        self, tracks: list[Track], now: float, threshold: float, min_seconds: float
    ) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float]]:
        observed: dict[tuple[int, str], float] = {}
        cleared: dict[tuple[int, str], float] = {}
        present: set[int] = set()
        for track in tracks:
            present.add(track.id)
            motion = self._motions.setdefault(track.id, _TrackMotion())
            height = max(1e-6, track.bbox.y2 - track.bbox.y1)
            sample = MotionSample(
                now,
                (track.bbox.x1 + track.bbox.x2) / 2,
                (track.bbox.y1 + track.bbox.y2) / 2,
                height,
            )
            if motion.samples and now <= motion.samples[-1].time + 1e-6:
                continue
            motion.samples.append(sample)
            cutoff = now - self.history_seconds
            while motion.samples and motion.samples[0].time < cutoff:
                motion.samples.popleft()
            speed = self._speed(motion.samples)
            key = (track.id, "running")
            if speed is None or height < self.min_height:
                # A broken measurement window must not let an old fast_since
                # accumulate credit and fire the moment the window recovers.
                motion.fast_since = None
                continue
            motion.last_speed = speed
            if speed >= threshold:
                if motion.fast_since is None:
                    motion.fast_since = now
                if now - motion.fast_since >= min_seconds:
                    confidence = min(1.0, 0.5 + 0.5 * (speed - threshold) / threshold)
                    observed[key] = round(confidence, 3)
            else:
                motion.fast_since = None
                if speed < threshold * 0.8:
                    cleared[key] = min(1.0, speed / max(threshold, 1e-6))
        for track_id in list(self._motions):
            if track_id not in present:
                motion = self._motions[track_id]
                if not motion.samples or now - motion.samples[-1].time > self.history_seconds * 2:
                    del self._motions[track_id]
        return observed, cleared

    def _speed(self, samples: deque[MotionSample]) -> float | None:
        if len(samples) < self.min_samples:
            return None
        span = samples[-1].time - samples[0].time
        if span < self.min_span_seconds:
            return None
        path = 0.0
        previous = samples[0]
        for current in islice(samples, 1, None):
            path += math.hypot(current.x - previous.x, current.y - previous.y)
            previous = current
        net = math.hypot(samples[-1].x - samples[0].x, samples[-1].y - samples[0].y)
        if path <= 1e-9 or net / path < self.straightness:
            return None
        mean_height = sum(s.height for s in samples) / len(samples)
        return net / span / max(mean_height, 1e-6)

    def diagnostics(self) -> dict[int, float]:
        return {key: round(motion.last_speed, 3) for key, motion in self._motions.items()}

    def observation_diagnostics(self):
        """Sampling eligibility, independent of whether speed exceeds the rule."""
        return {
            key: {
                "samples": len(motion.samples),
                "span_seconds": round(motion.samples[-1].time - motion.samples[0].time, 3)
                if motion.samples
                else 0,
                "history_ready": len(motion.samples) >= self.min_samples
                and motion.samples[-1].time - motion.samples[0].time >= self.min_span_seconds,
                "speed_person_heights_per_second": round(motion.last_speed, 3),
            }
            for key, motion in self._motions.items()
        }
