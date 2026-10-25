export type CommandState = {
  phase: "idle" | "executing" | "failed";
  error: string | null;
};
export type CommandEvent =
  | { type: "BEGIN" }
  | { type: "DONE" }
  | { type: "FAIL"; message: string }
  | { type: "DISMISS" };
export const initialCommand: CommandState = { phase: "idle", error: null };
export function commandTransition(
  state: CommandState,
  event: CommandEvent,
): CommandState {
  if (event.type === "BEGIN" && state.phase !== "executing")
    return { phase: "executing", error: null };
  if (event.type === "DONE" && state.phase === "executing")
    return initialCommand;
  if (event.type === "FAIL" && state.phase === "executing")
    return { phase: "failed", error: event.message };
  if (event.type === "DISMISS" && state.phase === "failed")
    return initialCommand;
  throw new Error(`Invalid command transition: ${state.phase} → ${event.type}`);
}
