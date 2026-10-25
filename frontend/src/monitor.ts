import type { Camera, Detection, Rule, Track } from "./types";

export type WorkerStatus = "violation" | "pending" | "ok" | "unknown";
export type ScreenStatus =
  | { kind: "idle" }
  | { kind: "calm"; workers: number }
  | {
      kind: "alert";
      level: "active" | "pending";
      type: string;
      trackId: number;
      elapsed: number;
      confirm: number;
    };

const VIOLATION = new Set(["no_helmet", "no_vest", "no_mask", "smoking"]);

/** Operator-facing status of one tracked worker. */
export function workerStatus(
  track: Track,
  rules: Rule[],
): WorkerStatus {
  if (track.violations.length > 0) return "violation";
  if (rules.some((r) => r.track_id === track.id)) return "pending";
  const states = [track.ppe["helmet"], track.ppe["vest"]];
  if (states.every((s) => s === "present")) return "ok";
  return "unknown";
}

/** Violators first, then pending, then the rest by id (stable order). */
export function sortTracks(tracks: Track[], rules: Rule[]): Track[] {
  const rank = (t: Track) =>
    workerStatus(t, rules) === "violation"
      ? 0
      : workerStatus(t, rules) === "pending"
        ? 1
        : 2;
  return [...tracks].sort((a, b) => rank(a) - rank(b) || a.id - b.id);
}

/**
 * Detections worth showing: PPE classes only (persons live in worker cards),
 * weak ones only when attached to a track. Kills background flicker without
 * hiding anything attributable to a worker.
 */
export function visibleDetections(detections: Detection[]): Detection[] {
  return detections.filter(
    (d) =>
      d.class_name !== "person" && (!d.weak || d.track_hint !== null),
  );
}

/** Violation-relevant detections first (red cards on top). */
export function sortDetections(detections: Detection[]): Detection[] {
  return [...detections].sort(
    (a, b) =>
      Number(VIOLATION.has(b.class_name)) - Number(VIOLATION.has(a.class_name)),
  );
}

/** Top banner state: worst active rule wins, else worst pending, else calm/idle. */
export function screenStatus(
  running: boolean,
  tracks: Track[],
  rules: Rule[],
): ScreenStatus {  if (!running) return { kind: "idle" };
  const actives = rules.filter((r) => r.phase === "active");
  if (actives.length > 0) {
    const top = [...actives].sort((a, b) => b.elapsed_seconds - a.elapsed_seconds)[0];
    return {
      kind: "alert",
      level: "active",
      type: top.type,
      trackId: top.track_id,
      elapsed: top.elapsed_seconds,
      confirm: top.confirm_seconds,
    };
  }
  const pendings = rules.filter((r) => r.phase !== "active");
  if (pendings.length > 0) {
    const top = [...pendings].sort(
      (a, b) =>
        b.elapsed_seconds / b.confirm_seconds - a.elapsed_seconds / a.confirm_seconds,
    )[0];
    return {
      kind: "alert",
      level: "pending",
      type: top.type,
      trackId: top.track_id,
      elapsed: top.elapsed_seconds,
      confirm: top.confirm_seconds,
    };
  }
  return { kind: "calm", workers: tracks.length };
}

export type CameraAlert = {
  cameraId: string;
  cameraName: string;
  level: "active" | "pending";
  type: string;
  trackId: number;
  elapsed: number;
  confirm: number;
};

function worstRule(
  rules: Rule[],
): { rule: Rule; level: "active" | "pending" } | null {
  const actives = rules.filter((r) => r.phase === "active");
  if (actives.length > 0) {
    const top = [...actives].sort((a, b) => b.elapsed_seconds - a.elapsed_seconds)[0];
    return { rule: top, level: "active" };
  }
  const pendings = rules.filter((r) => r.phase !== "active");
  if (pendings.length > 0) {
    const top = [...pendings].sort(
      (a, b) =>
        b.elapsed_seconds / b.confirm_seconds - a.elapsed_seconds / a.confirm_seconds,
    )[0];
    return { rule: top, level: "pending" };
  }
  return null;
}

/** Worst alert of one camera (active beats pending), or null when clean. */
export function cameraAlert(camera: Camera): CameraAlert | null {
  const rules = camera.latest?.rules ?? [];
  const hit = worstRule(rules);
  if (!hit) return null;
  return {
    cameraId: camera.id,
    cameraName: camera.name,
    level: hit.level,
    type: hit.rule.type,
    trackId: hit.rule.track_id,
    elapsed: hit.rule.elapsed_seconds,
    confirm: hit.rule.confirm_seconds,
  };
}

/** Worst alert across all cameras EXCEPT the selected one (or all, if none). */
export function globalAlert(
  cameras: Camera[],
  selectedId: string,
): CameraAlert | null {
  let top: CameraAlert | null = null;
  for (const camera of cameras) {
    if (camera.id === selectedId) continue;
    const alert = cameraAlert(camera);
    if (!alert) continue;
    if (
      !top ||
      (alert.level === "active" && top.level !== "active") ||
      (alert.level === top.level && alert.elapsed > top.elapsed)
    ) {
      top = alert;
    }
  }
  return top;
}
