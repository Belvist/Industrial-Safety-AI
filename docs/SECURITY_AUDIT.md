# Security audit (sprint `feature/security-model-quality`)

Scope: local single-operator service, LAN demo mode. Every item verified in
code + regression test, not claimed.

## S1. Demo token leakage — FIXED

- `system/info` stripped any `?token=` from `public_url`, covered by
  `test_system_info_never_leaks_api_token` (even misconfigured env).
- QR token comes from the operator browser URL, never from an API response.
- `run.py --lan` prints no token; uvicorn `--no-access-log` while the token
  gate is active (queries would land in access logs).
- Residual (documented): token lives in browser URLs (required for `<img>`
  auth); same-origin Referer only. LAN mode = trusted network only.

## S2. Audit results

| Check | Result |
|---|---|
| Unauthenticated GET/POST/PUT with token set | 401, tested (`test_lan.py`) |
| Snapshots/history without token | 401, tested (frame.jpg + snapshot) |
| Camera session hijacking | `X-Session-Id` must match live session, else 409 (tested) |
| RTSP/SSRF scheme | allowlist `rtsp/rtmp/http/https`, else 422 (`StartSource`) |
| Path traversal uploads | uuid filenames, no user paths; snapshot `Path(...).name` |
| Upload limits | video 200MB + cleanup, frame JPEG 5MB, empty rejected |
| Queues/memory | frames deque(4), notifier Queue(100)→`failed`, 429 backpressure |
| CORS/Origin/TrustedHost | loopback + configured LAN only; evil origin 403 (tested) |
| Tokens in logs | never logged (only incident id); bot URL excluded (tested) |
| Snapshot accumulation | `SNAPSHOT_RETENTION_DAYS` purge + orphan sweep on startup (tested) |
| SQLite | file-local, default umask; active rows clamped to RESOLVED/INTERRUPTED |

Open residual risks: token-holder in LAN can drive any camera (demo trust);
no brute-force throttling on token; SQLite file readable by local users.
All acceptable for trusted-network demo, not for a pilot — see P1 notes.

## Phone pairing flow (Sprint 1)

```text
operator (master token) ──POST /pairing {camera_id}──▶ code (TTL 5 min, one-time)
operator screen shows QR: <public_url>/connect?pair=CODE
phone ──POST /pairing/exchange {code}──▶ scoped token (camera_id, TTL 12 h)
phone ──X-API-Token──▶ start/stop/frame/frame.jpg of ITS camera only
camera stop ──▶ tokens for that camera revoked
```

Phone token permissions: GET/POST on `/api/v1/cameras/{own}/start|stop|frame|
frame.jpg|video` only. Denied: other cameras, PUT settings, incidents/history,
snapshots of others (snapshot route has no camera scope — denied entirely),
pairing management, system changes. Images load via authorized fetch + Blob
URL (no bearer in URLs, logs or Referer).
