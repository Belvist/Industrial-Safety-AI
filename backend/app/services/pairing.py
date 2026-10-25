"""One-time phone pairing for LAN demo mode.

Flow: authorized operator creates a short-lived pairing code for one camera.
The QR encodes `<public_url>/connect?pair=CODE` (no secrets leak through the
open API: the code itself is the secret, shown only on the operator screen).
The phone exchanges the code ONCE for a scoped phone token that authorizes
only its own camera. Stopping that camera revokes the token.

Single-process MVP: in-memory store with a lock, TTL sweep on every access.
No Redis, no extra service.
"""

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field


def token_owner_id(token: str) -> str:
    """Stable non-secret owner marker for a phone token."""
    return "phone:" + hashlib.sha256(token.encode()).hexdigest()[:16]


@dataclass
class PairingCode:
    code: str
    camera_id: str
    expires_at: float
    used: bool = False


@dataclass
class PhoneToken:
    token: str
    camera_id: str
    expires_at: float
    created_at: float = field(default_factory=time.time)


class PairingStore:
    def __init__(self, code_ttl_seconds: int = 300, token_ttl_seconds: int = 12 * 3600):
        self.code_ttl = code_ttl_seconds
        self.token_ttl = token_ttl_seconds
        self.codes: dict[str, PairingCode] = {}
        self.tokens: dict[str, PhoneToken] = {}
        self._lock = threading.Lock()

    def _sweep(self) -> None:
        now = time.monotonic()
        self.codes = {
            code: entry
            for code, entry in self.codes.items()
            if not entry.used and entry.expires_at > now
        }
        self.tokens = {
            token: entry
            for token, entry in self.tokens.items()
            if entry.expires_at > now
        }

    def create(self, camera_id: str) -> tuple[str, int]:
        """Mint a one-time pairing code. Returns (code, expires_in_seconds)."""
        with self._lock:
            self._sweep()
            code = secrets.token_urlsafe(12)
            self.codes[code] = PairingCode(
                code, camera_id, time.monotonic() + self.code_ttl
            )
            return code, self.code_ttl

    def bind(self, code: str, camera_id: str) -> bool:
        """Re-point a live code at a camera (new-camera flow)."""
        with self._lock:
            entry = self.codes.get(code)
            if entry is None or entry.used:
                return False
            entry.camera_id = camera_id
            return True

    def exchange(self, code: str) -> tuple[str, PhoneToken | None]:
        """Single-use exchange. Returns (status, token)."""
        with self._lock:
            entry = self.codes.get(code)
            if entry is None or entry.used:
                self.codes.pop(code, None)
                self._sweep()
                return "invalid", None
            if entry.expires_at <= time.monotonic():
                self.codes.pop(code, None)
                self._sweep()
                return "expired", None
            entry.used = True
            token = PhoneToken(
                secrets.token_urlsafe(24),
                entry.camera_id,
                time.monotonic() + self.token_ttl,
            )
            self.tokens[token.token] = token
            self._sweep()
            return "ok", token

    def camera_for(self, token: str) -> str | None:
        """Camera scope of a live phone token, else None."""
        with self._lock:
            self._sweep()
            entry = self.tokens.get(token)
            return entry.camera_id if entry else None

    def revoke_camera(self, camera_id: str) -> int:
        """Drop all codes/tokens bound to a camera (session end)."""
        with self._lock:
            before = len(self.codes) + len(self.tokens)
            self.codes = {c: e for c, e in self.codes.items() if e.camera_id != camera_id}
            self.tokens = {t: e for t, e in self.tokens.items() if e.camera_id != camera_id}
            return before - (len(self.codes) + len(self.tokens))

    def cancel(self, code: str) -> bool:
        with self._lock:
            return self.codes.pop(code, None) is not None
