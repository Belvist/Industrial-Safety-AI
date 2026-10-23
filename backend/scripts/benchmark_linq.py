#!/usr/bin/env python3
"""Authorized, bounded LinQ throughput test on real decoded video frames.

This is saturated file throughput, not a claim of realtime camera/UI FPS.
No broker settings, tokens or video paths are included in the report.
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.services.metrics import Metrics  # noqa: E402
from app.services.runtime import Runtime  # noqa: E402


def run_level(video, level, seconds):
    with tempfile.TemporaryDirectory(prefix="linq-benchmark-") as folder:
        cfg = Settings(
            linq_max_in_flight=level,
            inference_provider="linq",
            data_dir=folder,
            database_url=f"sqlite:///{folder}/db.sqlite",
            telegram_enabled=False,
        )
        if not (cfg.linq_amqp_host or cfg.linq_endpoint):
            raise RuntimeError("LINQ_AMQP_HOST is not configured; no hardware benchmark ran.")
        runtime = Runtime(cfg)
        provider = runtime.provider
        capture = cv2.VideoCapture(str(video))
        pool = ThreadPoolExecutor(max_workers=level)
        pending = deque()
        cam = next(iter(runtime.sessions.values()))
        try:
            if provider.mode != "linq_remote" or provider._person is None:
                raise RuntimeError(
                    "Requires remote TPU PPE and loaded CPU person model; no fallback."
                )
            if not capture.isOpened():
                raise RuntimeError("Cannot decode input video")
            # Real frames warm both independent branches before starting the clock.
            for _ in range(3):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Need at least three readable video frames")
                provider.predict_sync(frame)
            if not provider.worker_verified():
                raise RuntimeError("Worker has not verified successful remote inference")
            runtime.start(cam.id, "browser")
            provider.metrics = Metrics()
            cam.metrics = Metrics()
            cpu_start = time.process_time()
            started = time.monotonic()
            deadline = started + seconds
            errors, peak = 0, 0
            while time.monotonic() < deadline:
                if len(pending) < level:
                    with cam.metrics.stage("capture_decode_ms"):
                        ok, frame = capture.read()
                        if not ok:
                            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("Video loop could not be decoded")
                    captured = time.monotonic()
                    cam.metrics.mark("capture")
                    # Admission order and original timestamp travel with each Future.
                    pending.append((pool.submit(runtime._infer, frame, cam.metrics), captured))
                    peak = max(peak, len(pending))
                if pending and pending[0][0].done():
                    future, captured = pending.popleft()
                    try:
                        frame, detections, infer_ms = future.result()
                        with cam.lock:
                            runtime._commit(cam, frame, detections, captured, captured, infer_ms)
                    except Exception:
                        errors += 1
                else:
                    time.sleep(0.001)
            elapsed = time.monotonic() - started
            metrics = cam.metrics.snapshot()
            remote = provider.metrics.snapshot()
            analytics = metrics["rates"].get("analytics", {}).get("window_count", 0)
            tpu = remote["rates"].get("tpu", {}).get("window_count", 0)
            return {
                "input_mode": "saturated_file_loop",
                "in_flight": level,
                "duration_seconds": elapsed,
                "analytics_frames": analytics,
                "tpu_replies": tpu,
                "analytics_fps": analytics / elapsed,
                "tpu_fps": tpu / elapsed,
                "cpu_percent_one_core": (time.process_time() - cpu_start) / elapsed * 100,
                "failures": errors,
                "pending_at_window_end": len(pending),
                "peak_in_flight": peak,
                "metrics": metrics,
                "remote_metrics": remote,
                "worker_verified": provider.worker_verified(),
                "hardware_verified": provider.accelerator_confirmed(),
                "ui_fps": None,
                "broker_queue_latency_ms": None,
                "five_fps_demonstrated": elapsed >= 60
                and analytics >= 300
                and analytics / elapsed >= 5
                and tpu / elapsed >= 5
                and errors == 0,
            }
        finally:
            for future, _ in pending:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            capture.release()
            runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--approved-max",
        type=int,
        required=True,
        help="Concurrency explicitly permitted by the resource owner (1..12)",
    )
    parser.add_argument("--levels", default="1,2,4,6,8,12")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.approved_max <= 12 or args.seconds < 60:
        parser.error("approved-max must be 1..12 and seconds >= 60")
    levels = [int(x) for x in args.levels.split(",")]
    if any(x < 1 or x > 12 for x in levels):
        parser.error("levels must be in 1..12")
    allowed = sorted(set(x for x in levels if x <= args.approved_max))
    if not allowed:
        parser.error("No levels within the approved limit")
    report = {
        "video_sha256": hashlib.file_digest(args.video.open("rb"), "sha256").hexdigest(),
        "approved_max": args.approved_max,
        "results": [],
        "status": "running",
    }
    try:
        for level in allowed:
            result = run_level(args.video, level, args.seconds)
            report["results"].append(result)
            print(
                json.dumps(
                    {k: result[k] for k in ["in_flight", "analytics_fps", "tpu_fps", "failures"]}
                )
            )
            # Don't escalate load after a failed level on a shared resource.
            if result["failures"]:
                report["status"] = "stopped_after_failures"
                break
        else:
            report["status"] = "completed"
    except Exception as exc:
        report["status"] = "blocked"
        # Avoid exception strings from libraries containing broker credentials.
        report["error_type"] = type(exc).__name__
        report["reason"] = (
            "Check configured LinQ access, CPU person weights and input video. No fallback was used."
        )
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
