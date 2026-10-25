"""Bounded, thread-safe stage samples and actual rolling event rates."""

import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.samples = defaultdict(lambda: deque(maxlen=4096))
        self.events = defaultdict(lambda: deque(maxlen=18000))
        self.totals = defaultdict(int)

    def observe(self, name, value):
        with self.lock:
            self.samples[name].append(float(value))

    def mark(self, name, count=1):
        with self.lock:
            self.events[name].append((time.monotonic(), count))
            self.totals[name] += count

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, (time.perf_counter() - started) * 1000)

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            duration = min(60.0, max(0.001, now - self.started))
            rates = {}
            for name, events in self.events.items():
                while events and events[0][0] < now - 60:
                    events.popleft()
                count = sum(n for _, n in events)
                rates[name] = {
                    "fps": round(count / duration, 3),
                    "window_count": count,
                    "total": self.totals[name],
                }
            stages = {}
            for name, values in self.samples.items():
                p50, p95, p99 = np.percentile(values, [50, 95, 99])
                stages[name] = {
                    "samples": len(values),
                    "p50": round(float(p50), 3),
                    "p95": round(float(p95), 3),
                    "p99": round(float(p99), 3),
                    "max": round(float(max(values)), 3),
                }
            return {"window_seconds": round(duration, 3), "rates": rates, "stages": stages}
