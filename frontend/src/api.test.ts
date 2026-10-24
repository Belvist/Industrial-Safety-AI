import { describe, it, expect, vi, afterEach } from "vitest";
import { api, authHeaders, errorText, pairingUrl, ApiError, bootstrapOperatorAuth, saveOperatorToken } from "./api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("pairing auth", () => {
  it("sends no auth header without any token", () => {
    expect(authHeaders()).toEqual({});
  });
  it("prefers the page URL token, then the stored phone token", () => {
    vi.stubGlobal(
      "location",
      new URL("https://192.168.1.194:8000/?token=master"),
    );
    vi.stubGlobal("localStorage", {
      getItem: () => "phone-token",
    });
    expect(authHeaders()).toEqual({ "X-API-Token": "master" });
  });
  it("uses the stored phone token on the phone page", () => {
    vi.stubGlobal("location", new URL("https://192.168.1.194:8000/phone"));
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => (key === "phone_token" ? "phone-token" : null),
    });
    expect(authHeaders()).toEqual({ "X-API-Token": "phone-token" });
  });
  it("does not use a phone token on the operator dashboard", () => {
    vi.stubGlobal("location", new URL("https://192.168.1.194:8000/"));
    vi.stubGlobal("sessionStorage", { getItem: () => null });
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => (key === "phone_token" ? "phone-token" : null),
    });
    expect(authHeaders()).toEqual({});
  });
  it("builds the QR pairing URL without leaking other secrets", () => {
    expect(pairingUrl("https://192.168.1.194:8000/", "abc123")).toBe(
      "https://192.168.1.194:8000/connect?pair=abc123",
    );
  });
  it("api() authorizes through headers, never the URL", async () => {
    vi.stubGlobal(
      "location",
      new URL("https://192.168.1.194:8000/?token=master"),
    );
    vi.stubGlobal("localStorage", { getItem: () => null });
    const calls: Array<[string, RequestInit]> = [];
    vi.stubGlobal("fetch", async (url: string, init: RequestInit) => {
      calls.push([url, init]);
      return { ok: true, json: async () => ({}) } as Response;
    });
    await api("/cameras");
    expect(calls[0][0]).toBe("/api/v1/cameras");
    expect((calls[0][1].headers as Record<string, string>)["X-API-Token"]).toBe(
      "master",
    );
  });
  it("explains 401 with pairing guidance", () => {
    expect(errorText(new ApiError("Нужен токен доступа.", 401))).toContain(
      "введите ключ",
    );
    expect(errorText(new ApiError("x", 500))).toBe("x");
  });
});


describe("operator LAN authentication", () => {
  it("remembers the operator key for this tab and clears the URL", () => {
    const items: Record<string, string> = {};
    let replaced = "";
    vi.stubGlobal("location", new URL("https://192.168.1.194:8000/?token=secret-operator"));
    vi.stubGlobal("sessionStorage", {
      getItem: (key: string) => items[key] ?? null,
      setItem: (key: string, value: string) => { items[key] = value; },
    });
    vi.stubGlobal("history", {
      state: null,
      replaceState: (_state: unknown, _unused: string, url: string) => { replaced = url; },
    });
    bootstrapOperatorAuth();
    expect(items.operator_token).toBe("secret-operator");
    expect(replaced).toBe("/");
    expect(authHeaders()).toEqual({ "X-API-Token": "secret-operator" });
  });
  it("allows manual operator unlock when IP address was opened without token", () => {
    const items: Record<string, string> = {};
    vi.stubGlobal("location", new URL("https://192.168.1.194:8000/"));
    vi.stubGlobal("sessionStorage", {
      getItem: (key: string) => items[key] ?? null,
      setItem: (key: string, value: string) => { items[key] = value; },
    });
    vi.stubGlobal("localStorage", { getItem: () => null });
    saveOperatorToken("  secret-operator  ");
    expect(authHeaders()).toEqual({ "X-API-Token": "secret-operator" });
  });
  it("never grants the operator token to the phone capture route", () => {
    vi.stubGlobal("location", new URL("https://192.168.1.194:8000/phone"));
    vi.stubGlobal("sessionStorage", { getItem: () => "master-secret" });
    vi.stubGlobal("localStorage", { getItem: (key: string) => key === "phone_token" ? "phone-only" : null });
    expect(authHeaders()).toEqual({ "X-API-Token": "phone-only" });
  });
});
