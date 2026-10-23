import { observeClient } from "./clientMetrics";
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
/** Operator access is confined to this browser tab. A paired phone keeps
 * its separate scoped token and can never inherit operator privileges. */
export function saveOperatorToken(token: string): void {
  if (typeof sessionStorage !== "undefined") {
    sessionStorage.setItem("operator_token", token.trim());
  }
}

/** Convert the launcher URL credential into a tab-scoped header credential.
 * Removing it from the address bar also prevents copying it by accident. */
export function bootstrapOperatorAuth(): void {
  if (typeof location === "undefined" || location.pathname !== "/") return;
  const url = new URL(location.href);
  const token = url.searchParams.get("token");
  if (!token) return;
  saveOperatorToken(token);
  url.searchParams.delete("token");
  if (typeof history !== "undefined") {
    history.replaceState(
      history.state,
      "",
      url.pathname + url.search + url.hash,
    );
  }
}
function accessToken(): string | null {
  if (typeof location === "undefined") return null;
  if (location.pathname === "/phone" || location.pathname === "/connect") {
    return typeof localStorage !== "undefined"
      ? localStorage.getItem("phone_token")
      : null;
  }
  return (
    new URLSearchParams(location.search).get("token") ||
    (typeof sessionStorage !== "undefined"
      ? sessionStorage.getItem("operator_token")
      : null) ||
    (typeof localStorage !== "undefined"
      ? localStorage.getItem("phone_token")
      : null)
  );
}
/** Auth header for API calls (preferred over query strings). */
export function authHeaders(): Record<string, string> {
  const token = accessToken();
  return token ? { "X-API-Token": token } : {};
}
/** Pairing URL encoded into the operator QR (code is the credential). */
export function pairingUrl(base: string, code: string): string {
  return base.replace(/\/$/, "") + "/connect?pair=" + encodeURIComponent(code);
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const started = performance.now();
  try {
    const response = await fetch("/api/v1" + path, {
      ...options,
      headers: { ...(options.headers || {}), ...authHeaders() },
    });
    if (!response.ok) {
      const data = await response
        .json()
        .catch(() => ({ detail: "Сервер вернул ошибку" }));
      const message =
        typeof data.detail === "string"
          ? data.detail
          : Array.isArray(data.detail)
            ? data.detail.map((e: { msg: string }) => e.msg).join("; ")
            : "Ошибка запроса";
      throw new ApiError(message, response.status);
    }
    return await response.json();
  } finally {
    observeClient("api_roundtrip_ms", performance.now() - started);
  }
}
export const json = (value: unknown, method = "POST") => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(value),
});
export const errorText = (value: unknown) => {
  if (value instanceof ApiError && value.status === 401)
    return (
      "Нет доступа (401): введите ключ оператора в форме входа " +
      "или подключите телефон через QR."
    );
  return value instanceof Error ? value.message : String(value);
};
export const stopPath = (cameraId: string, sessionId?: string | null) =>
  `/cameras/${cameraId}/stop${sessionId ? "?session_id=" + encodeURIComponent(sessionId) : ""}`;
