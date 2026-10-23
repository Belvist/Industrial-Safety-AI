"""Full Runtime benchmark: single V4-cloud model on Mac/CoreML, >=10 min.

Drives the REAL app path (Runtime.start 'video' -> _source_loop ->
process(): inference + tracker + assign + TemporalEngine + annotate +
JPEG encode; SQLite + snapshot save on ACTIVE) with an isolated data dir
and tmp database. Nothing in production settings or weights is modified:
the script builds its own Settings object pointing at models/ppe-v4cloud.onnx
in single-model mode (use_separate_person_model=False).

Input: concatenated real footage (~12 min) built from footage_v4/yt_eval
clips (ffmpeg, documented concat list). Transport = video file, NOT a live
camera — stated plainly in the report (decode replaces capture).

Measures: wall time, frames processed (session.sequence), achieved FPS,
per-frame end-to-end latency p50/p95 (timed wrapper around process()),
CPU% + RSS sampling, active ONNX providers, incidents + snapshots written.

Usage (repo root, .venv):
    .venv/bin/python training/bench_runtime.py --source footage_v4/yt_eval_12min.mp4 \\
        --minutes 12 --out training/bench_runtime
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))


def sample_proc(pid: int):
    try:
        out = subprocess.run(['ps', '-o', '%cpu,rss', '-p', str(pid)],
                             capture_output=True, text=True, check=False).stdout
        lines = out.strip().splitlines()
        cpu, rss = lines[1].split()
        return float(cpu), int(rss) // 1024
    except (ValueError, IndexError, subprocess.SubprocessError):
        return 0.0, 0


def main() -> None:
    from app.core.config import Settings
    from app.services.runtime import Runtime

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--minutes', type=float, default=12.0)
    ap.add_argument('--target-fps', type=float, default=5.0,
                    help='pacing cap of the video source loop (5.0 = live-like; '
                         '30 = unpaced max-throughput proof)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='bench_rt_'))

    # Model + person mode come from ambient settings (.env): the bench measures
    # the ACTIVE prod configuration, never a hardcoded model.
    settings = Settings(
        data_dir=str(work / 'data'),
        database_url=f'sqlite:///{work}/bench.db',
        onnx_providers='CoreMLExecutionProvider',
        target_fps=args.target_fps,
    )
    rt = Runtime(settings)
    cams = list(rt.sessions)
    assert len(cams) == 1, cams
    cam = cams[0]
    print('providers:', list(rt.provider.active_providers), flush=True)

    lat, cpu, rss = [], [], []
    stop_flag = threading.Event()

    def sampler():
        while not stop_flag.wait(2.0):
            c, r = sample_proc(os.getpid())
            cpu.append(c)
            rss.append(r)

    orig_process = rt.process

    def timed_process(session, frame, timestamp=None):
        # Count only successful calls: a stop-time race raises inside
        # process(), and a finally-based timer would lie by one.
        t0 = time.perf_counter()
        out = orig_process(session, frame, timestamp)
        lat.append((time.perf_counter() - t0) * 1000)
        return out

    rt.process = timed_process  # type: ignore[method-assign]
    rt.start(cam, 'video', args.source)
    threading.Thread(target=sampler, daemon=True).start()
    deadline = time.monotonic() + args.minutes * 60
    last_seq, seq = 0, 0
    while time.monotonic() < deadline:
        time.sleep(5)
        sess = rt.session(cam)
        seq = sess.sequence
        st = sess.status
        print(f't={time.monotonic():.0f}s seq={seq} state={st} '
              f'lat_n={len(lat)}', flush=True)
        if st in {'ended', 'error', 'stopped'} and seq == last_seq:
            break
        last_seq = seq
    wall = time.monotonic()
    try:
        rt.stop(cam, revoke_tokens=False)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the bench
        print('stop:', exc, flush=True)
    stop_flag.set()
    wall = time.monotonic() - (deadline - args.minutes * 60)

    lat_sorted = sorted(lat)
    seq = rt.session(cam).sequence  # fresh read AFTER stop: the poll-loop
    # value is stale by construction (thread keeps processing between polls)
    # Single source of truth: timed process() calls == processed frames.
    # Source total (cv2 frame count) minus processed = skipped by the loop.
    import cv2 as _cv2
    _cap = _cv2.VideoCapture(str(args.source))
    source_frames = int(_cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
    _cap.release()
    assert len(lat_sorted) == seq, \
        f'counter drift: sequence={seq} timed_calls={len(lat_sorted)}'
    def pct(q):
        return round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * q))], 1) \
            if lat_sorted else 0.0
    incidents_resp = rt.storage.incidents()
    incidents = incidents_resp["items"] if isinstance(incidents_resp, dict) \
        else list(incidents_resp)
    snaps = list((work / 'data' / 'snapshots').glob('*'))
    snap_ok = sum(1 for i in incidents
                  if i.get('snapshot') and (work / 'data' / 'snapshots' / i['snapshot']).is_file())
    import hashlib
    _h = hashlib.sha256()
    with open(ROOT / settings.model_path, 'rb') as _f:
        for _chunk in iter(lambda: _f.read(1 << 20), b''):
            _h.update(_chunk)
    rep = {
        'source': args.source, 'wall_seconds': round(wall, 1),
        'model': {'path': settings.model_path, 'sha256': _h.hexdigest(),
                  'person_model': settings.person_model_path,
                  'separate_person': settings.use_separate_person_model},
        'settings': {'thresholds': dict(settings.class_thresholds),
                     'roi': settings.ppe_roi_mode,
                     'target_fps': settings.target_fps,
                     'confirm_s': settings.incident_confirm_seconds,
                     'clear_s': settings.incident_clear_seconds,
                     'window': settings.temporal_window,
                     'required': settings.temporal_required},
        'frames_processed': len(lat_sorted), 'fps_achieved': round(seq / wall, 2) if wall else 0,
        'source_frames': source_frames, 'skipped_frames': source_frames - len(lat_sorted),
        'latency_ms': {'p50': pct(0.5), 'p95': pct(0.95), 'n': len(lat_sorted)},
        'cpu_pct': {'mean': round(sum(cpu) / len(cpu), 1) if cpu else 0,
                    'max': round(max(cpu), 1) if cpu else 0},
        'rss_mb': {'mean': round(sum(rss) / len(rss)) if rss else 0,
                   'max': max(rss) if rss else 0},
        'active_providers': list(rt.provider.active_providers),
        'incidents': len(incidents), 'snapshots': len(snaps),
        'snapshots_verified': snap_ok,
        'transport': 'video file (decode), not a live camera',
    }
    (out / 'runtime.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1), flush=True)
    try:
        rt.close()
    except Exception as exc:  # noqa: BLE001 - bench must report even if close fails
        print('close:', exc, flush=True)


if __name__ == '__main__':
    main()
