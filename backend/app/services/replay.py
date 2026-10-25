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


def encode_replay(frames, fps, path):
    """Encode the existing source JPEGs once; no intermediate lossy MPEG-4.

    Both decoder and encoder threads are bounded. The caller publishes the
    staged file atomically only after this function completes.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-threads",
                "1",
                "-f",
                "image2pipe",
                "-c:v",
                "mjpeg",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            input=b"".join(frame[1] for frame in frames),
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    import cv2
    import numpy as np

    first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError("Unreadable evidence frame")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("H.264 evidence needs FFmpeg/libx264 or a native avc1 encoder")
    try:
        for frame in frames:
            image = cv2.imdecode(np.frombuffer(frame[1], np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                writer.write(image)
    finally:
        writer.release()
