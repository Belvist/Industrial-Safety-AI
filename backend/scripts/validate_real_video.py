#!/usr/bin/env python3
"""Real video -> actual ONNX/LinQ -> tracker/radar -> SQLite -> MP4.

No cached detections or scripted risks are accepted as inference input.
Source/media time and measured wall throughput are reported independently.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import Settings
from app.services.runtime import Runtime
from app.services.schemas import CameraConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--warmup", type=float, default=5)
    p.add_argument(
        "--window-seconds",
        type=float,
        default=0,
        help="Measure an actual wall-clock window after warmup",
    )
    p.add_argument("--provider", choices=["local_onnx", "linq"], default="local_onnx")
    p.add_argument("--cameras", type=int, default=1, choices=range(1, 5))
    args = p.parse_args()
    if args.seconds <= 0 or args.warmup < 0 or args.window_seconds < 0:
        p.error("Durations must be positive (warmup/window may be zero).")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new empty output directory; acceptance data must not mix runs.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = CameraConfig(**json.loads(args.config.read_text()))
    cfg = Settings(
        inference_provider=args.provider,
        data_dir=str(args.output / "runtime"),
        database_url=f"sqlite:///{(args.output / 'runtime' / 'safety.db').resolve()}",
        telegram_enabled=False,
        file_realtime_motion=True,
    )
    report = {
        "source_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "config": config.model_dump(),
        "provider_requested": args.provider,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "cameras": [],
    }
    if args.provider == "linq" and not (cfg.linq_amqp_host or cfg.linq_endpoint):
        report.update(
            status="BLOCKED",
            reason="No configured LinQ endpoint; no fallback was run.",
            worker_verified=False,
        )
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, ensure_ascii=False))
        return 2
    runtime = Runtime(cfg)
    if runtime.model_error:
        raise RuntimeError(runtime.model_error)
    cameras = []
    try:
        initial = next(iter(runtime.sessions.values()))
        initial.config = config
        runtime.storage.save_camera(config, initial.id)
        cameras.append(initial)
        if args.cameras > 1:
            from app.services.runtime import CameraSession

            for i in range(1, args.cameras):
                camera_config = config.model_copy(update={"name": config.name + f" #{i + 1}"})
                cid = runtime.storage.save_camera(camera_config)
                runtime.sessions[cid] = CameraSession(cid, camera_config)
                cameras.append(runtime.sessions[cid])
        observations = {c.id: [] for c in cameras}
        sequences = {c.id: 0 for c in cameras}
        stopped = set()
        windows = {}
        snapshots = {}
        measurements = {}
        started = time.monotonic()
        for c in cameras:
            runtime.start(c.id, "video", str(args.video.resolve()))
        deadline = started + max(60, args.seconds * 3, args.warmup + args.window_seconds + 30)
        while len(stopped) < len(cameras) and time.monotonic() < deadline:
            for c in cameras:
                if c.id in stopped:
                    continue
                view = c.summary()
                frame = view.get("latest")
                if frame and frame["sequence"] != sequences[c.id]:
                    sequences[c.id] = frame["sequence"]
                    observations[c.id].append(frame)
                    clock = time.monotonic()
                    if (
                        args.window_seconds
                        and c.id not in windows
                        and clock - started >= args.warmup
                    ):
                        windows[c.id] = (clock, dict(c.metrics.snapshot()["rates"]))
                    done = (
                        args.window_seconds
                        and c.id in windows
                        and clock - windows[c.id][0] >= args.window_seconds
                    ) or (not args.window_seconds and frame["source_timestamp"] >= args.seconds)
                    if done:
                        snapshots[c.id] = c.metrics.snapshot()
                        if c.id in windows:
                            window_start, initial_rates = windows[c.id]
                            duration = clock - window_start
                            counts = {
                                k: snapshots[c.id]["rates"].get(k, {}).get("total", 0)
                                - initial_rates.get(k, {}).get("total", 0)
                                for k in (
                                    "person",
                                    "radar",
                                    "analytics",
                                    "tpu",
                                    "ppe",
                                    "capture",
                                    "dropped",
                                )
                            }
                            measurements[c.id] = {
                                "duration_seconds": duration,
                                "counts": counts,
                                "fps": {k: v / duration for k, v in counts.items()},
                            }
                        runtime.stop(c.id)
                        stopped.add(c.id)
                if view["status"] in ["ended", "error"]:
                    stopped.add(c.id)
            time.sleep(0.01)
        elapsed = time.monotonic() - started
        for c in cameras:
            frames = observations[c.id]
            risk = runtime.storage.risk_statistics({c.id})
            incidents = runtime.storage.incidents(camera_ids={c.id})["items"]
            with runtime.storage.connection() as db:
                events = [
                    dict(r)
                    for r in db.execute(
                        "SELECT * FROM observation_events WHERE camera_id=? ORDER BY id", (c.id,)
                    )
                ]
            report["cameras"].append(
                {
                    "camera_id": c.id,
                    "frames_observed": len(frames),
                    "wall_seconds": elapsed,
                    "metrics": snapshots.get(c.id, c.metrics.snapshot()),
                    "measurement_window": measurements.get(c.id),
                    "last_source_seconds": frames[-1]["source_timestamp"] if frames else None,
                    "terminal_state": c.status,
                    "error": c.error,
                    "events": events,
                    "incidents": incidents,
                    "statistics": risk,
                    "ppe_observations": sum(
                        any(v != "unknown" for v in t["ppe"].values())
                        for f in frames
                        for t in f["tracks"]
                    ),
                    "tracks_issued": c.tracker.next_id - 1,
                    "max_people": max((len(f["tracks"]) for f in frames), default=0),
                    "running_history_ready_frames": sum(
                        any(t["history_ready"] for t in f.get("motion_history", {}).values())
                        for f in frames
                    ),
                    "radar_history_ready_frames": sum(
                        any(
                            len(t["trajectory"]) >= 3
                            for t in (f.get("radar") or {}).get("scene", {}).get("tracks", [])
                        )
                        for f in frames
                    ),
                    "maximum_data_age_ms": max(
                        (f.get("data_age_ms", 0) for f in frames), default=0
                    ),
                    "unique_ppe_source_frames": len(
                        {
                            f["ppe"]["source_frame_id"]
                            for f in frames
                            if f.get("ppe") and f["ppe"].get("source_frame_id") is not None
                        }
                    ),
                }
            )
            (args.output / f"{c.id}-frames.json").write_text(
                json.dumps(frames, indent=2, ensure_ascii=False)
            )
        report.update(
            status="MEASURED"
            if len(snapshots) == len(cameras) and all(c.status != "error" for c in cameras)
            else "INCOMPLETE",
            worker_verified=getattr(runtime.provider, "worker_verified", lambda: False)(),
            provider=runtime.provider.status(),
            independent_motion=runtime.independent_motion,
        )
    finally:
        runtime.close()
    evidence = args.output / "runtime" / "evidence"
    report["replay"] = [
        {
            "file": f.name,
            "bytes": f.stat().st_size,
            "ffprobe": json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_entries",
                        "stream=codec_name,duration,nb_frames",
                        "-of",
                        "json",
                        str(f),
                    ],
                    text=True,
                )
            ),
        }
        for f in sorted(evidence.glob("*.mp4"))
    ]
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {
                "status": report["status"],
                "worker_verified": report["worker_verified"],
                "cameras": [
                    {
                        k: c[k]
                        for k in [
                            "frames_observed",
                            "wall_seconds",
                            "tracks_issued",
                            "ppe_observations",
                        ]
                    }
                    for c in report["cameras"]
                ],
                "replay_count": len(report["replay"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
