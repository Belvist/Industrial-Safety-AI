import { describe, expect, it } from "vitest";
import { observation } from "./model";
import type { Camera, Result } from "../types";

const camera = { status: "running" } as Camera;
const frame = { rules: [], tracks: [], data_age_ms: 0 } as unknown as Result;
const risk = (level: "CRITICAL" | "WARNING" | "UNKNOWN") => ({
  ...frame,
  radar: { track_risks: { 1: level }, events: [] },
});

describe("operator observation status", () => {
  it("does not present a previous result as live after loss of connection or source", () => {
    expect(observation(camera, frame, false).tone).not.toBe("live");
    for (const status of [
      "stopped",
      "ended",
      "error",
      "stale",
      "starting",
    ] as const)
      expect(observation({ ...camera, status }, frame, true).tone).not.toBe(
        "live",
      );
    expect(
      observation(camera, { ...frame, data_age_ms: 5001 }, true).tone,
    ).not.toBe("live");
    expect(observation(camera, null, true).tone).not.toBe("live");
  });
  it("keeps a critical radar state urgent before incident confirmation", () => {
    expect(observation(camera, risk("CRITICAL"), true).tone).toBe("danger");
    expect(observation(camera, risk("WARNING"), true).title).toContain(
      "приближается",
    );
    expect(observation(camera, risk("UNKNOWN"), true).tone).toBe("warning");
  });
  it("keeps a confirmed violation visible while clearing is being checked", () => {
    const rules: Result["rules"] = [
      {
        track_id: 1,
        type: "no_helmet",
        phase: "clearing",
        confidence: 0.9,
        elapsed_seconds: 1,
        confirm_seconds: 2,
      },
    ];
    expect(observation(camera, { ...frame, rules }, true).tone).toBe("danger");
    expect(
      observation(
        camera,
        { ...frame, rules: [{ ...rules[0], phase: "pending" }] },
        true,
      ).tone,
    ).toBe("warning");
  });
  it("describes an empty observed frame without claiming workplace safety", () => {
    const state = observation(camera, frame, true);
    expect(state.tone).toBe("live");
    expect(state.description).toContain("пока не обнаружено");
    expect(state.description).not.toMatch(/безопасно|без нарушений/i);
  });
});
