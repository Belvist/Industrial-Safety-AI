import type { Info } from "./types";

const LINQ_ACTIVE = new Set(["hardware_verified"]);

/** True only after confirmed inference (verified) or hardware proof. */
export function isLinqActive(info: Info | null): boolean {
  return (
    info?.inference_provider === "linq" &&
    LINQ_ACTIVE.has(info?.inference?.readiness ?? "")
  );
}

/**
 * Honest inference label. "LinQ Active" appears only after verification;
 * a configured-but-unproven endpoint is named as such, never as active.
 */
export function inferenceLabel(info: Info | null): string {
  if (!info) return "Inference: —";
  const inf = info.inference;
  const active = inf?.active_providers?.[0];
  const execution = active ? ` · ${active}` : " · CPU";
  if (inf?.readiness === "inference_verified" && info.inference_provider === "linq")
    return "Inference: Remote inference verified · LinQ hardware unconfirmed";
  if (isLinqActive(info))
    return `Inference: LinQ Active · ${inf?.mode}${active ? ` · ${active}` : ""}`;
  if (info.inference_provider === "linq") {
    if (inf?.readiness === "failed") return "Inference: LinQ unreachable";
    if (inf && (inf.readiness === "configured" || inf.readiness === "connecting"))
      return "Inference: LinQ configured · awaiting verification";
    if (inf?.readiness === "reachable")
      return "Inference: LinQ reachable · inference not proven";
    return `Inference: Local ONNX fallback${execution}`;
  }
  return `Inference: Local ONNX${execution}`;
}

/** Second line under the badge when LinQ is configured but inactive. */
export function inferenceSub(info: Info | null): string | null {
  if (!info || info.inference_provider !== "linq" || isLinqActive(info)) return null;
  if (info.inference?.readiness === "failed")
    return "endpoint unreachable — check LINQ_ENDPOINT";
  return "LinQ configured, accelerator inactive";
}

/**
 * Auto-diagnostic: names hardware acceleration available on this machine but
 * not used by the active session. Never claims it is active.
 */
export function accelerationHint(info: Info | null): string | null {
  const available = info?.available_providers ?? [];
  const active = info?.inference?.active_providers ?? [];
  const idle = available.filter(
    (p) => p !== "CPUExecutionProvider" && !active.includes(p),
  );
  if (!idle.length) return null;
  return `Доступно ускорение: ${idle.join(", ")} (не активно)`;
}
