export type CapturePhase =
  | "idle"
  | "permission"
  | "connecting"
  | "capturing"
  | "submitting"
  | "waiting"
  | "stopping"
  | "failed";
export type CaptureEvent =
  | "START"
  | "PERMITTED"
  | "CONNECTED"
  | "CAPTURED"
  | "RESULT"
  | "NEXT"
  | "STOP"
  | "STOPPED"
  | "FAIL";
export type CaptureState = {
  phase: CapturePhase;
  cameraId: string | null;
  error: string | null;
};
export const initialCapture: CaptureState = {
  phase: "idle",
  cameraId: null,
  error: null,
};
const transitions: Record<
  CapturePhase,
  Partial<Record<CaptureEvent, CapturePhase>>
> = {
  idle: { START: "permission" },
  permission: { PERMITTED: "connecting", STOP: "stopping", FAIL: "failed" },
  connecting: { CONNECTED: "capturing", STOP: "stopping", FAIL: "failed" },
  capturing: { CAPTURED: "submitting", STOP: "stopping", FAIL: "failed" },
  submitting: {
    CAPTURED: "submitting",
    RESULT: "waiting",
    STOP: "stopping",
    FAIL: "failed",
  },
  waiting: { NEXT: "capturing", STOP: "stopping", FAIL: "failed" },
  stopping: { STOPPED: "idle", FAIL: "failed" },
  failed: { START: "permission", STOP: "stopping", STOPPED: "idle" },
};
export function transition(
  state: CaptureState,
  event: CaptureEvent,
  cameraId?: string,
  error?: string,
): CaptureState {
  const phase = transitions[state.phase][event];
  if (!phase)
    throw new Error(`Invalid capture transition: ${state.phase} → ${event}`);
  return {
    phase,
    cameraId: phase === "idle" ? null : (cameraId ?? state.cameraId),
    error: phase === "failed" ? (error ?? "Камера недоступна") : null,
  };
}
// Immutable snapshots support React's useSyncExternalStore; state changes are synchronous.
export class CaptureMachine {
  private state = initialCapture;
  private listeners = new Set<() => void>();
  getSnapshot = () => this.state;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };
  send(event: CaptureEvent, cameraId?: string, error?: string) {
    this.state = transition(this.state, event, cameraId, error);
    for (const listener of this.listeners) listener();
  }
}
