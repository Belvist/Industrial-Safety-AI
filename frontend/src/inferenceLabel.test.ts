import { describe, it, expect } from "vitest";
import { accelerationHint, inferenceLabel, inferenceSub, isLinqActive } from "./inferenceLabel";
import type { Info } from "./types";

const base: Info = {
  model_ready: true,
  model_error: null,
  telegram_ready: false,
  target_fps: 5,
  inference_provider: "local_onnx",
  inference: null,
  available_providers: ["CPUExecutionProvider"],
  public_url: null,
  stats: { total: 0, active: 0, today: 0, false_positive: 0 },
};

describe("inferenceLabel", () => {
  it("handles missing info", () => {
    expect(inferenceLabel(null)).toBe("Inference: —");
    expect(inferenceSub(null)).toBeNull();
  });
  it("names the real local execution provider, never LinQ", () => {
    const info: Info = {
      ...base,
      inference: {
        provider: "local_onnx",
        mode: "local_onnx",
        active_providers: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
      },
    };
    expect(inferenceLabel(info)).toBe(
      "Inference: Local ONNX · CoreMLExecutionProvider",
    );
    expect(inferenceLabel(info)).not.toContain("LinQ");
    expect(isLinqActive(info)).toBe(false);
    expect(inferenceSub(info)).toBeNull();
  });
  it("falls back to CPU wording without telemetry", () => {
    expect(inferenceLabel(base)).toBe("Inference: Local ONNX · CPU");
  });
  it("shows LinQ Active only after verification", () => {
    const remote: Info = {
      ...base,
      inference_provider: "linq",
      inference: { provider: "linq", mode: "linq_remote", readiness: "inference_verified" },
    };
    expect(isLinqActive(remote)).toBe(false);
    expect(inferenceLabel(remote)).toContain("Remote inference verified");
    expect(inferenceLabel(remote)).toContain("hardware unconfirmed");
    expect(inferenceLabel(remote)).not.toContain("LinQ Active");
  });
  it("health-only endpoint is reachable, never active", () => {
    const info: Info = {
      ...base,
      inference_provider: "linq",
      inference: { provider: "linq", mode: "linq_remote", readiness: "reachable" },
    };
    expect(isLinqActive(info)).toBe(false);
    expect(inferenceLabel(info)).toBe("Inference: LinQ reachable · inference not proven");
  });
  it("hardware proof marks LinQ Active", () => {
    const info: Info = {
      ...base,
      inference_provider: "linq",
      inference: {
        provider: "linq",
        mode: "linq_local",
        readiness: "hardware_verified",
        active_providers: ["LinQExecutionProvider"],
      },
    };
    expect(isLinqActive(info)).toBe(true);
    expect(inferenceLabel(info)).toContain("LinQ Active");
  });
  it("configured-but-unverified endpoint is never shown as active", () => {
    const info: Info = {
      ...base,
      inference_provider: "linq",
      inference: { provider: "linq", mode: "linq_remote", readiness: "configured" },
    };
    expect(isLinqActive(info)).toBe(false);
    expect(inferenceLabel(info)).toBe("Inference: LinQ configured · awaiting verification");
    expect(inferenceSub(info)).toBe("LinQ configured, accelerator inactive");
  });
  it("failed endpoint is shown as unreachable", () => {
    const info: Info = {
      ...base,
      inference_provider: "linq",
      inference: { provider: "linq", mode: "linq_remote", readiness: "failed" },
    };
    expect(isLinqActive(info)).toBe(false);
    expect(inferenceLabel(info)).toBe("Inference: LinQ unreachable");
    expect(inferenceSub(info)).toContain("unreachable");
  });
  it("configured-but-inactive LinQ is a local fallback, not active LinQ", () => {
    const info: Info = {
      ...base,
      inference_provider: "linq",
      inference: {
        provider: "linq",
        mode: "local_onnx",
        readiness: "fallback",
        active_providers: ["CPUExecutionProvider"],
      },
    };
    expect(isLinqActive(info)).toBe(false);
    const label = inferenceLabel(info);
    expect(label).toBe("Inference: Local ONNX fallback · CPUExecutionProvider");
    expect(label).not.toContain("LinQ Active");
    expect(inferenceSub(info)).toBe("LinQ configured, accelerator inactive");
  });
  it("advertises idle hardware acceleration without claiming it active", () => {
    expect(accelerationHint(base)).toBeNull();
    const hint = accelerationHint({
      ...base,
      available_providers: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
      inference: {
        provider: "local_onnx",
        mode: "local_onnx",
        active_providers: ["CPUExecutionProvider"],
      },
    });
    expect(hint).toContain("CoreMLExecutionProvider");
    expect(hint).toContain("не активно");
    expect(
      accelerationHint({
        ...base,
        available_providers: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        inference: {
          provider: "local_onnx",
          mode: "local_onnx",
          active_providers: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        },
      }),
    ).toBeNull();
  });
});
