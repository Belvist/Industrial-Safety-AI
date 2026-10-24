"""Same instrumentation, real inference and unchanged images for baseline and candidate."""

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--repo", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--video", type=Path, required=True)
p.add_argument("--cameras", type=int, default=1)
p.add_argument("--ppe", action="store_true")
p.add_argument("--target-fps", type=float, default=5)
p.add_argument("--window", type=float, default=60)
p.add_argument("--warmup", type=float, default=5)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--provider", choices=["local_onnx", "linq"], default="local_onnx")
args = p.parse_args()
if args.window <= 0 or args.warmup < 0 or not 1 <= args.cameras <= 8:
    p.error("Positive window, nonnegative warmup and 1..8 cameras required")


def cgroup_cpu():
    """cgroup v2 CPU stats; unavailable on macOS/Windows returns empty."""
    try:
        return dict(
            (key, int(value))
            for key, value in (
                line.split() for line in Path("/sys/fs/cgroup/cpu.stat").read_text().splitlines()
            )
        )
    except (OSError, ValueError):
        return {}


sys.path.insert(0, str(args.repo / "backend"))
from app.core.config import Settings  # noqa: E402
from app.services.runtime import CameraSession, Runtime  # noqa: E402
from app.services.schemas import CameraConfig  # noqa: E402

args.output.mkdir(parents=True, exist_ok=False)
cfg = Settings(
    _env_file=None,
    data_dir=str(args.output / "runtime"),
    database_url=f"sqlite:///{args.output}/runtime/s.db",
    inference_provider=args.provider,
    telegram_enabled=False,
    inference_threads=args.threads,
    target_fps=args.target_fps,
)
if args.provider == "linq" and not (cfg.linq_amqp_host or cfg.linq_endpoint):
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED",
                "reason": "No configured LinQ endpoint; no inference/fallback performed.",
                "worker_verified": False,
            },
            indent=2,
        )
    )
    raise SystemExit(2)
r = Runtime(cfg)
assert not r.model_error, r.model_error
config = CameraConfig(
    **json.loads((args.repo / "docs/acceptance/configs/worker-zone-motion.json").read_text())
)
config.helmet = config.vest = args.ppe
cams = [next(iter(r.sessions.values()))]
cams[0].config = config
for i in range(1, args.cameras):
    cc = config.model_copy(update={"name": config.name + f" #{i + 1}"})
    cid = r.storage.save_camera(cc)
    r.sessions[cid] = CameraSession(cid, cc)
    cams.append(r.sessions[cid])
lock = threading.Lock()
samples = []
events = []
frames = []


def instrument(metrics, name):
    observe, mark = metrics.observe, metrics.mark

    def obs(stage, value):
        stamp = time.monotonic()
        with lock:
            samples.append((stamp, name, stage, float(value)))
        return observe(stage, value)

    def event(stage, count=1):
        stamp = time.monotonic()
        with lock:
            events.append((stamp, name, stage, count))
        return mark(stage, count)

    metrics.observe = obs
    metrics.mark = event


instrument(r.metrics, "global")
connection = r.storage.connection


@contextlib.contextmanager
def timed_connection():
    start = time.perf_counter()
    with connection() as db:
        yield db
    with lock:
        samples.append(
            (
                time.monotonic(),
                "global",
                "sqlite_connection_ms",
                (time.perf_counter() - start) * 1000,
            )
        )


r.storage.connection = timed_connection
commit = r._commit
start = time.monotonic()


def traced_commit(cam, *a):
    result = commit(cam, *a)
    with lock:
        frames.append(
            {
                "wall": time.monotonic(),
                "camera_id": cam.id,
                "sequence": result["sequence"],
                "source_timestamp": result["source_timestamp"],
                "age_ms": result["data_age_ms"],
                "media_lag_ms": max(
                    0, (time.monotonic() - start - result["source_timestamp"]) * 1000
                ),
                "people": len(result["tracks"]),
                "ppe": result.get("ppe"),
                "risks": result.get("radar", {}).get("track_risks", {}),
            }
        )
    return result


r._commit = traced_commit
try:
    for c in cams:
        r.start(c.id, "video", str(args.video))
        instrument(c.metrics, c.id)
    time.sleep(args.warmup)
    window_start = time.monotonic()
    cpu_start = time.process_time()
    group_start = cgroup_cpu()
    deadline = window_start + args.window
    while time.monotonic() < deadline:
        time.sleep(0.01)
    end = time.monotonic()
    group_end = cgroup_cpu()
    cpu = (time.process_time() - cpu_start) / (end - window_start) * 100
    with lock:
        window_events = [x for x in events if window_start <= x[0] < end]
        window_samples = [x for x in samples if window_start <= x[0] < end]
        window_frames = [x for x in frames if window_start <= x["wall"] < end]

    def percentiles(values):
        if not values:
            return None
        return {
            "samples": len(values),
            **dict(
                zip(
                    ("p50", "p95", "p99"),
                    map(float, np.percentile(values, [50, 95, 99])),
                    strict=True,
                )
            ),
            "max": max(values),
            "mean": float(np.mean(values)),
            "sum": sum(values),
        }

    rows = []
    for c in cams:
        counts = {
            name: sum(x[3] for x in window_events if x[1] == c.id and x[2] == name)
            for name in (
                "capture",
                "sampled_out",
                "person",
                "radar",
                "analytics",
                "ppe",
                "ppe_applied",
                "tpu",
                "dropped",
                "ppe_stale",
                "ppe_identity_rejected",
            )
        }
        ff = [x for x in window_frames if x["camera_id"] == c.id]
        rows.append(
            {
                "camera_id": c.id,
                "counts": counts,
                "fps": {k: v / (end - window_start) for k, v in counts.items()},
                "age_ms": percentiles([x["age_ms"] for x in ff]),
                "media_lag_ms": percentiles([x["media_lag_ms"] for x in ff]),
                "max_people": max((x["people"] for x in ff), default=0),
                "stages": {
                    name: percentiles(
                        [x[3] for x in window_samples if x[1] == c.id and x[2] == name]
                    )
                    for name in sorted({x[2] for x in window_samples if x[1] == c.id})
                },
            }
        )
    terminal = [{"camera_id": c.id, "status": c.status, "error": c.error} for c in cams]
    for c in cams:
        r.stop(c.id)
    r.close()
    report = {
        "status": "MEASURED" if all(c["status"] == "running" for c in terminal) else "INCOMPLETE",
        "terminal_before_stop": terminal,
        "worker_verified": getattr(r.provider, "worker_verified", lambda: False)(),
        "git_sha": subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(args.repo), "status", "--porcelain"], text=True
            ).strip()
        ),
        "source_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "duration_seconds": end - window_start,
        "warmup_seconds": args.warmup,
        "target_fps": args.target_fps,
        "ppe_enabled": args.ppe,
        "provider": r.provider.status(),
        "cpu_percent_one_core": cpu,
        "cgroup_cpu_percent_one_core": (
            (group_end["usage_usec"] - group_start["usage_usec"])
            / ((end - window_start) * 10000)
            if "usage_usec" in group_start and "usage_usec" in group_end
            else None
        ),
        "cgroup_throttled_periods": (
            group_end.get("nr_throttled", 0) - group_start.get("nr_throttled", 0)
            if group_start and group_end
            else None
        ),
        "cpu_quota": (
            Path("/sys/fs/cgroup/cpu.max").read_text().strip()
            if Path("/sys/fs/cgroup/cpu.max").exists()
            else "not available (non-Linux host)"
        ),
        "cameras": rows,
        "global_stages": {
            name: percentiles([x[3] for x in window_samples if x[1] == "global" and x[2] == name])
            for name in sorted({x[2] for x in window_samples if x[1] == "global"})
        },
        "incidents": r.storage.incidents()["items"],
        "all_events": [],
        "late_shutdown_stages": r.metrics.snapshot(),
        "note": "Rates and stage percentiles measured in same post-warmup wall window. File backpressure/media lag explicitly reported. No TPU inference in local mode.",
    }
    with r.storage.connection() as db:
        report["all_events"] = [
            dict(row) for row in db.execute("SELECT * FROM observation_events ORDER BY id")
        ]
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (args.output / "frames.json").write_text(json.dumps(frames, ensure_ascii=False))
    print(
        json.dumps(
            {"output": str(args.output), "fps": [x["fps"]["analytics"] for x in rows], "cpu": cpu},
            ensure_ascii=False,
        )
    )
finally:
    if not r.shutdown.is_set():
        r.close()
