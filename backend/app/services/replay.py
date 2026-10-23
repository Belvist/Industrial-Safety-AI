"""Resample timestamped evidence without speeding up slow analytics."""

import bisect
import math


def replay_frames(frames, fps, event_time=None):
    if not frames or any(len(frame) < 4 for frame in frames):
        return frames  # compatibility with old, uniformly sampled buffers
    times = [frame[3] for frame in frames]
    first, last = times[0], times[-1]
    if event_time is not None:
        first, last = max(first, event_time - 3), min(last, event_time + 1)
    else:
        first = max(first, last - 6)
    if last < first:
        return []
    output = []
    for index in range(math.ceil((last - first) * fps) + 1):
        timestamp = min(last, first + index / fps)
        source = max(0, bisect.bisect_right(times, timestamp) - 1)
        output.append(frames[source])
    return output
