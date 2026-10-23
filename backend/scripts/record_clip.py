"""Record a raw clip from a camera or screen for PPE evaluation footage.

Each recording becomes one scene dir later: move the mp4 + write LICENSE.json
(origin/self-recorded, consent true) + a scene_id shared by all clips of one
shoot (see training/build_ppe_v4.py). Files stay local (footage_v4/ is
git-ignored); never commit raw footage.

Usage:
    .venv/bin/python backend/scripts/record_clip.py --source 0 --seconds 180 \\
        --out footage_v4/raw/sceneA_cam1.mp4
    .venv/bin/python backend/scripts/record_clip.py --source path/to/file.mp4 \\
        --seconds 120 --out footage_v4/raw/sceneB_cam1.mp4   # re-cut a segment
Keys while recording: q = stop early. A meta sidecar (.json) is written next
to the mp4 (source, fps, size, started_at_utc).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import cv2


def open_source(source: str):
    try:
        index = int(source)
        return cv2.VideoCapture(index), None
    except ValueError:
        return cv2.VideoCapture(str(source)), str(source)


def measure_rate(stamps: list[float]) -> float | None:
    """Frames/sec from wall-clock stamps (pure, unit-tested).

    Irregular delivery (USB jitter, exposure stalls) is averaged out over
    the whole calibration window instead of trusting the first interval.
    """
    if len(stamps) < 2 or stamps[-1] <= stamps[0]:
        return None
    return (len(stamps) - 1) / (stamps[-1] - stamps[0])


def calibrate_camera(cap, seconds: float = 3.0, max_frames: int = 90):
    """Measure the true delivery rate; returns (fps or None, frames_read).

    Falls back to None (caller uses requested fps + warns) when the device
    yields nothing decodable during calibration.
    """
    import time as _time

    stamps, kept = [], []
    t0 = _time.monotonic()
    while len(kept) < max_frames and _time.monotonic() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        stamps.append(_time.monotonic() - t0)
        kept.append(frame)
    return measure_rate(stamps), kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, help='camera index (0) or video file')
    ap.add_argument('--seconds', type=float, default=180.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=25.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap, origin = open_source(args.source)
    if not cap.isOpened():
        raise SystemExit(f'Cannot open source: {args.source}')
    src_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    is_camera = origin is None
    calibrated: bool | None = None
    if is_camera:
        # Physical cameras lie about fps (or report 0/30 for a 10 fps feed).
        # Measure first: the writer MUST run at the true delivery rate or the
        # mp4 plays back at the wrong speed despite correct sidecar stamps.
        measured, _kept = calibrate_camera(cap)
        if measured:
            src_fps, calibrated = measured, True
        else:
            calibrated = False
    target_fps = min(src_fps, args.fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out), fourcc, target_fps, (w, h))
    if not writer.isOpened():
        raise SystemExit('VideoWriter failed (codec missing?)')
    started = datetime.datetime.now(datetime.UTC).isoformat()
    # Drift-free sampling: media timestamps from the source clock, never
    # round()-stepped counters. Camera sources (unknown fps) fall back to
    # wall-clock stamps per written frame.
    is_camera = origin is None
    n, idx, next_ts, stamps, t0 = 0, 0, 0.0, [], time.monotonic()
    print('recording — q quits early', flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        media_t = idx / src_fps if not is_camera and src_fps > 0 else None
        if media_t is None:
            take = True
        else:
            take = media_t + 1e-6 >= next_ts
        clock_now = time.monotonic() - t0
        if take:
            if (frame.shape[1], frame.shape[0]) != (w, h):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
            stamps.append(round(media_t if media_t is not None else clock_now, 3))
            n += 1
            next_ts += 1.0 / target_fps
        idx += 1
        if is_camera:
            limit = clock_now
        else:
            limit = media_t if media_t is not None else clock_now
        if limit >= args.seconds:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    meta = {'source': args.source if origin is None else origin,
            'started_at_utc': started, 'fps': target_fps,
            'requested_fps': args.fps,
            'measured_fps': round(src_fps, 2) if is_camera else None,
            'calibrated': calibrated,
            'size': [w, h], 'frames': n, 'stamps_sec': stamps,
            'duration_media_sec': round(stamps[-1], 3) if stamps else 0.0,
            'note': 'EVAL ONLY unless LICENSE.json + consent say otherwise. '
                    'Camera recordings start after ~3 s calibration lead-in.'}
    out.with_suffix('.json').write_text(json.dumps(meta, indent=1))
    print(f'saved {out} ({n} frames) + sidecar', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise SystemExit('Pass --source 0 --seconds 180 --out footage_v4/raw/sceneA_cam1.mp4')
    main()
