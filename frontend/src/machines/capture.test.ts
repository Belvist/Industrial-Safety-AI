import { describe, it, expect } from "vitest";
import { CaptureMachine, initialCapture, transition } from "./capture";
import { commandTransition, initialCommand } from "./command";

describe("camera client machine", () => {
  it("requires permission and connection before submitting a frame", () => {
    const machine = new CaptureMachine();
    machine.send("START", "camera-a");
    expect(machine.getSnapshot().phase).toBe("permission");
    expect(() => machine.send("CAPTURED")).toThrow(
      "Invalid capture transition",
    );
    for (const event of [
      "PERMITTED",
      "CONNECTED",
      "CAPTURED",
      "RESULT",
      "NEXT",
    ] as const)
      machine.send(event);
    expect(machine.getSnapshot()).toEqual({
      phase: "capturing",
      cameraId: "camera-a",
      error: null,
    });
    machine.send("CAPTURED");
    machine.send("STOP");
    expect(() => machine.send("RESULT")).toThrow();
    machine.send("STOPPED");
    expect(machine.getSnapshot()).toEqual(initialCapture);
  });
  it("can cancel permission or retry after a denied camera", () => {
    const machine = new CaptureMachine();
    machine.send("START", "a");
    machine.send("STOP");
    machine.send("STOPPED");
    machine.send("START", "b");
    machine.send("FAIL", undefined, "Permission denied");
    expect(machine.getSnapshot().error).toBe("Permission denied");
    machine.send("START", "b");
    expect(machine.getSnapshot().error).toBe(null);
  });
  it("publishes immutable snapshots and cleans subscriptions", () => {
    const machine = new CaptureMachine();
    let count = 0;
    const before = machine.getSnapshot();
    const unsubscribe = machine.subscribe(() => count++);
    machine.send("START", "a");
    unsubscribe();
    machine.send("PERMITTED");
    expect(count).toBe(1);
    expect(before).toEqual(initialCapture);
    expect(machine.getSnapshot()).not.toBe(before);
  });
});
describe("command machine", () => {
  it("rejects concurrent commands and supports recovery after failure", () => {
    const pending = commandTransition(initialCommand, { type: "BEGIN" });
    expect(() => commandTransition(pending, { type: "BEGIN" })).toThrow();
    const failed = commandTransition(pending, {
      type: "FAIL",
      message: "offline",
    });
    expect(failed.error).toBe("offline");
    expect(commandTransition(failed, { type: "DISMISS" })).toEqual(
      initialCommand,
    );
    expect(
      commandTransition(commandTransition(failed, { type: "BEGIN" }), {
        type: "DONE",
      }),
    ).toEqual(initialCommand);
  });
});
