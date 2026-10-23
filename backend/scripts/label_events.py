"""Keyboard labeling of PPE violation intervals (no JSON hand-editing).

Streams the clip (frames read on demand with a small cache — a 10-minute
video never sits fully in RAM) and records per-worker violation intervals
straight to events.json (the format evaluate_folder.py scores).

Keys:
    space  play / pause            , .  step -/+ 1 frame (when paused)
    1      violation type = no_helmet    2  violation type = no_vest
    w      next worker (w1..w4, wraps)  [  interval start at current frame
    ]      interval end (saves {type, worker, start, end, boxes})
    b      box mode: drag worker box with the mouse (stored per frame)
    u      undo last event           s  save events.json + quit
    q      quit WITHOUT saving

Output events.json: {"events": [{type, worker, start_frame, end_frame,
boxes: {frame: [x1,y1,x2,y2]}}]} at --fps numbering (default 5 to match
evaluate_folder.py). Boxes make worker matching verifiable; events without
them score as worker_unverified.

Requires a display (cv2.imshow). Headless fallback: extract frames and label
by filename ranges with --dump-frames, then write events.json by hand once.

Usage:
    .venv/bin/python backend/scripts/label_events.py --clip footage_v4/raw/sceneA_cam1.mp4 \\
        --out footage_v4/raw/sceneA_cam1_events.json --fps 5
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import cv2

TYPES = {'1': 'no_helmet', '2': 'no_vest'}
CACHE = 48


def display_to_norm(ax, ay, w0, h0):
    """Display pixels -> normalized coords with independent x/y scales.

    The preview resize is (min(w,1280) x min(h,720)), NOT uniform, so
    portrait video (e.g. 720x1280 shown as 720x720) needs both scales.
    """
    sx = min(w0, 1280) / w0
    sy = min(h0, 720) / h0
    return round(ax / sx / w0, 4), round(ay / sy / h0, 4)


class LazyFrames:
    """Seek-read clip frames with a small LRU cache (bounded RAM)."""

    def __init__(self, path: str, fps: float):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f'Cannot open clip: {path}')
        src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # Time-based selection (never round-stepped): frame closest at or
        # after each k/fps instant, so 25->7 fps is really 7.0, not 6.25.
        self.src_fps = src_fps
        positions, k = [], 0
        while True:
            idx = round(k * src_fps / fps)
            if idx >= total:
                break
            if not positions or idx != positions[-1]:
                positions.append(idx)
            k += 1
            if k > total + 2:
                break
        self.positions = positions or [0]
        self.cache: collections.OrderedDict[int, object] = collections.OrderedDict()
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    def __len__(self):
        return len(self.positions)

    def get(self, i: int):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i].copy()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.positions[i])
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit(f'Cannot decode frame {i}.')
        self.cache[i] = frame
        while len(self.cache) > CACHE:
            self.cache.popitem(last=False)
        return frame.copy()

    def close(self):
        self.cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--clip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--fps', type=float, default=5.0)
    ap.add_argument('--dump-frames', default=None,
                    help='headless fallback: dump every Nth frame to DIR and exit')
    args = ap.parse_args()

    frames = LazyFrames(str(args.clip), args.fps)
    if not len(frames):
        raise SystemExit('No frames decoded.')
    if args.dump_frames:
        d = Path(args.dump_frames)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(len(frames)):
            cv2.imwrite(str(d / f'f{i:05d}.jpg'), frames.get(i),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f'dumped {len(frames)} frames to {d} (label ranges by hand)', flush=True)
        return

    events, pos, playing = [], 0, False
    worker, vtype, start = 'w1', 'no_helmet', None
    boxes: dict[str, dict[int, list]] = {}
    wid = 1
    drag = None

    def on_mouse(event, x, y, _flags, _param):
        nonlocal drag
        if event == cv2.EVENT_LBUTTONDOWN:
            drag = [(x, y), (x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drag is not None:
            drag[1] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag is not None:
            (ax, ay), (bx, by) = drag
            h0, w0 = frames.h or 1, frames.w or 1
            x_lo, x_hi = sorted([ax, bx])
            y_lo, y_hi = sorted([ay, by])
            x1, y1 = display_to_norm(x_lo, y_lo, w0, h0)
            x2, y2 = display_to_norm(x_hi, y_hi, w0, h0)
            box = [x1, y1, x2, y2]
            boxes.setdefault(worker, {})[pos] = box
            print(f'box {worker} f{pos}: {box}', flush=True)
            drag = None

    cv2.namedWindow('label_events')
    cv2.setMouseCallback('label_events', on_mouse)
    print('space=play/pause 1/2=type w=worker [=start ]=end drag=worker-box '
          'u=undo s=save q=quit', flush=True)
    while True:
        fr = frames.get(pos)
        h, w = fr.shape[:2]
        for wb, wb_map in boxes.items():
            if pos in wb_map:
                x1, y1, x2, y2 = wb_map[pos]
                cv2.rectangle(fr, (int(x1 * w), int(y1 * h)),
                              (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)
                cv2.putText(fr, wb, (int(x1 * w), max(14, int(y1 * h) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        status = f'f{pos}/{len(frames)} {worker} {vtype} ' \
                 f'{"REC..." if start is not None else ""} events={len(events)}'
        cv2.putText(fr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        if start is not None:
            cv2.putText(fr, f'start={start}', (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        disp_w = min(w, 1280)
        shown = cv2.resize(fr, (disp_w, min(h, 720)))
        if drag is not None:
            cv2.rectangle(shown, drag[0], drag[1], (0, 255, 255), 1)
        cv2.imshow('label_events', shown)
        key = cv2.waitKey(30 if playing else 0) & 0xFF
        if key == ord('q'):
            print('quit WITHOUT saving', flush=True)
            break
        if key == ord(' '):
            playing = not playing
        elif key == ord(','):
            pos = max(0, pos - 1)
            playing = False
        elif key == ord('.'):
            pos = min(len(frames) - 1, pos + 1)
            playing = False
        elif chr(key) in TYPES:
            vtype = TYPES[chr(key)]
        elif key == ord('w'):
            wid = wid % 4 + 1
            worker = f'w{wid}'
        elif key == ord('['):
            start = pos
        elif key == ord(']'):
            if start is None:
                print('no start set', flush=True)
            else:
                ev_boxes = {str(f): b for f, b in sorted(boxes.get(worker, {}).items())
                            if start <= f <= pos}
                events.append({'type': vtype, 'worker': worker,
                               'start_frame': start, 'end_frame': pos,
                               'boxes': ev_boxes})
                print(f'event #{len(events)}: {vtype} {worker} {start}-{pos} '
                      f'boxes={len(ev_boxes)}', flush=True)
                start = None
        elif key == ord('u'):
            if events:
                print('undid', events.pop(), flush=True)
        elif key == ord('s'):
            Path(args.out).write_text(json.dumps({'events': events}, indent=1))
            print(f'saved {len(events)} events to {args.out}', flush=True)
            break
        if playing:
            pos += 1
            if pos >= len(frames):
                pos, playing = len(frames) - 1, False
    frames.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise SystemExit('Pass --clip <mp4> --out <events.json> [--fps 5]')
    main()
