# Focused pipeline and interface security review — 2026-10-23

Scope: changes from `8f581d6` to the concurrent pipeline/operator workspace.
This is a focused code and dependency review, not a penetration-test certificate.

| Finding | Change | Verification |
|---|---|---|
| Unauthenticated `system/info` used an unrestricted stats scope | Empty scope for unauthenticated/expired callers; provider internals and model error text are redacted | Public response contains zero private incident counts |
| Camera `owner` was accepted from configuration input | POST assigns no client owner; PUT preserves server ownership | Forged owner cannot reassign the camera |
| Evidence name was independent of requested camera | Require the generated camera filename prefix and a resolved evidence-directory path | Cross-camera filename rejected; traversal/symlink escape guarded |
| 5 MB compressed frame limit did not bound decoded pixels | Validate JPEG/PNG dimensions before decoding, cap at 16 Mpixels; decoder errors return 400 | Oversized PNG rejected before OpenCV allocation |
| Long inference held camera state lock | Separate bounded admission, independent inference and ordered commits | Stop/state reads remain available during a delayed prediction |
| Late results could affect restarted sessions/new rules | Session and rules generation guards, browser source IDs and ordered tickets | Stale-result and concurrent-frame regressions |
| Zone-wide deduplication could merge two workers | Removed identity-free camera/zone incident reuse | Two workers produce two separate incidents |
| Every Replay link used the first event's video | Select the clicked event's evidence URL | Per-event selection in the UI; authenticated fetches retained |
| New results could label an old JPEG | Carry image and inference payload together | Source-associated image state; browser E2E |
| Evidence could omit the event frame or speed up low-FPS data | Snapshot after including current frame; timestamp-based replay sampling; bound evidence queue | Slow-frame timing regression and actual H.264/ffprobe check |
| Known test-stack and CA-package advisories | Vitest 5.0.5, pytest 9.0.3, certifi 2026.10.19 | npm audit: 0; project Python closure audit: 0 |

API responses use `no-store`; responses also set `nosniff`, frame denial and
no-referrer headers. Operator token comparison is constant-time. Named polygons
all receive the geometry checks previously applied only to the legacy zone;
duplicate zone names are rejected. Duplicate schema fields were removed.

Retained boundaries and limits:

- Default local mode without an operator token is intentionally local/trusted;
  the launcher token/LAN origin setup remains necessary for remote access.
- An authenticated operator can configure private RTSP/HTTP camera addresses.
  This local product needs private-camera access; it is not an internet-hosted,
  multi-tenant arbitrary-URL service. Phone credentials still cannot start RTSP.
- Shared LinQ AMQP/vendored protocol has no authenticated independent hardware
  counters, remote task cancellation or broker queue-depth contract. The new
  exclusive reply queue and IDs do not replace organizer network isolation.
- A requested remote worker failure never silently falls back to CPU PPE.
  Existing explicitly configured local-development fallback is still labelled local.
- Actual models and confidence thresholds were not changed. Labelled event accuracy,
  physical phone checks and real remote performance were unavailable in this session.
- Native desktop signing, updater/installer security and public hosting are outside
  this web-interface change. FFmpeg build licenses must accompany any later bundle.

Dependency findings before fixes: npm reported 2 critical and 1 moderate advisories
in Vitest/transitive test packages; the Python project closure reported advisories
for certifi and pytest. The broader execution environment includes unrelated tools;
only the project's 41-package dependency closure was used for the scoped result.


## Operator workflow follow-up — 24 October 2026

- Review filtering uses parameterized SQL and the existing camera scope.
- CSV is operator-only. It shares journal validation, reads a single SQLite
  snapshot, caps export at 10,000 records and neutralizes formula-leading cells
  (including whitespace-prefixed formulas). Unauthenticated/scoped-phone export
  rejection and exports spanning more than one UI page are tested.
- Evidence uses authenticated blob fetches and revokes object URLs on cleanup.
  Missing media has an explicit retry/error state. H.264 actually plays in the
  isolated Chromium review test.
- Pairing-code expiry now counts down; expired QR codes disappear. Camera/source
  changes clear the displayed code. Operator tokens remain tab-scoped.
- A cancelled remote startup cleans up the acknowledged session by session ID;
  a failed history refresh does not falsely report an already committed save as
  failed. Unsaved settings are guarded on navigation and browser Back.

These checks do not establish a complete security audit or enterprise deployment
certification. The boundaries listed above remain in force.
