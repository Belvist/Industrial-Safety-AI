import { describe, expect, it } from "vitest";
import {
  screenStatus,
  sortDetections,
  sortTracks,
  visibleDetections,
  workerStatus,
} from "./monitor";
import type { Detection, Rule, Track } from "./types";

const track = (over: Partial<Track> & { id: number }): Track => ({
  bbox: { x1: 0, y1: 0, x2: 1, y2: 1 },
  confidence: 0.9,
  violations: [],
  ppe: {},
  ppe_confidence: {},
  ...over,
});

const det = (over: Partial<Detection> & { class_name: string }): Detection => ({
  confidence: 0.8,
  bbox: { x1: 0, y1: 0, x2: 1, y2: 1 },
  track_hint: null,
  weak: false,
  ...over,
});

const rule = (over: Partial<Rule>): Rule => ({
  track_id: 1,
  type: "no_helmet",
  phase: "pending",
  confidence: 0.5,
  elapsed_seconds: 1,
  confirm_seconds: 2,
  ...over,
});

describe("workerStatus", () => {
  it("violation beats pending beats ok", () => {
    expect(
      workerStatus(track({ id: 1, violations: ["no_helmet"] }), [rule({})]),
    ).toBe("violation");
    expect(workerStatus(track({ id: 1 }), [rule({})])).toBe("pending");
    expect(
      workerStatus(
        track({ id: 1, ppe: { helmet: "present", vest: "present" } }),
        [],
      ),
    ).toBe("ok");
    expect(workerStatus(track({ id: 1 }), [])).toBe("unknown");
  });
});

describe("sortTracks", () => {
  it("puts violators first, stable by id", () => {
    const sorted = sortTracks(
      [track({ id: 3 }), track({ id: 1, violations: ["no_vest"] }), track({ id: 2 })],
      [rule({ track_id: 2 })],
    ).map((t) => t.id);
    expect(sorted).toEqual([1, 2, 3]);
  });
});

describe("visibleDetections", () => {
  it("hides persons and unattached weak noise", () => {
    const out = visibleDetections([
      det({ class_name: "person" }),
      det({ class_name: "no_helmet", weak: true }),
      det({ class_name: "no_helmet", weak: true, track_hint: "2" }),
      det({ class_name: "vest" }),
    ]);
    expect(out.map((d) => [d.class_name, d.weak, d.track_hint])).toEqual([
      ["no_helmet", true, "2"],
      ["vest", false, null],
    ]);
  });
});

describe("sortDetections", () => {
  it("puts violations on top", () => {
    const out = sortDetections([
      det({ class_name: "helmet" }),
      det({ class_name: "no_vest" }),
    ]);
    expect(out[0].class_name).toBe("no_vest");
  });
});

describe("screenStatus", () => {
  it("idle when not running", () => {
    expect(screenStatus(false, [], []).kind).toBe("idle");
  });
  it("calm with worker count", () => {
    expect(screenStatus(true, [track({ id: 1 })], [] as Rule[])).toEqual({
      kind: "calm",
      workers: 1,
    });
  });
  it("active beats pending", () => {
    const s = screenStatus(true, [], [
      rule({ type: "no_vest", phase: "pending", elapsed_seconds: 1.9 }),
      rule({ type: "no_helmet", phase: "active", elapsed_seconds: 0.5 }),
    ]);
    expect(s).toMatchObject({ kind: "alert", level: "active", type: "no_helmet" });
  });
});

import { cameraAlert, globalAlert } from "./monitor";
import type { Camera } from "./types";

const cam = (over: Partial<Camera> & { id: string }): Camera => ({
  name: "K",
  status: "running",
  session_id: "s",
  state_version: 1,
  error: null,
  kind: "browser",
  latest: null,
  helmet: true,
  vest: true,
  mask: false,
  smoking: false,
  zone_enabled: false,
  zone: [],
  confirm_seconds: 2,
  clear_seconds: 2,
  temporal_window: 4,
  temporal_required: 3,
  ...over,
});

describe("cameraAlert", () => {
  it("null when clean, active beats pending", () => {
    expect(cameraAlert(cam({ id: "a" }))).toBeNull();
    const alert = cameraAlert(
      cam({
        id: "a",
        name: "Цех",
        latest: {
          session_id: "s",
          sequence: 1,
          frame_url: "u",
          tracks: [],
          detections: [],
          rules: [
            rule({ type: "no_vest", phase: "pending", track_id: 2 }),
            rule({ type: "no_helmet", phase: "active", track_id: 1 }),
          ],
          fps: 5,
          latency_ms: 40,
          active: 1,
          width: 1,
          height: 1,
          timestamp: "t",
        },
      }),
    );
    expect(alert).toMatchObject({
      cameraId: "a",
      cameraName: "Цех",
      level: "active",
      type: "no_helmet",
      trackId: 1,
    });
  });
});

describe("globalAlert", () => {
  it("skips the selected camera", () => {
    const mk = (id: string) =>
      cam({
        id,
        latest: {
          session_id: "s",
          sequence: 1,
          frame_url: "u",
          tracks: [],
          detections: [],
          rules: [rule({ track_id: 1 })],
          fps: 5,
          latency_ms: 40,
          active: 1,
          width: 1,
          height: 1,
          timestamp: "t",
        },
      });
    expect(globalAlert([mk("a"), mk("b")], "a")?.cameraId).toBe("b");
    expect(globalAlert([mk("a")], "a")).toBeNull();
  });
});
