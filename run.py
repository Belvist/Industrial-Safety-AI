"""One-command local setup and launch for Windows, macOS and Linux."""

import argparse
import hashlib
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command, cwd=ROOT):
    subprocess.run([str(p) for p in command], cwd=cwd, check=True)


def _is_private_lan(ip: str) -> bool:
    parts = ip.split(".")
    if len(parts) != 4 or not all(p.isdigit() for p in parts):
        return False
    a, b = int(parts[0]), int(parts[1])
    if a == 10 or (a == 172 and 16 <= b <= 31) or (a == 192 and b == 168):
        return ip != "127.0.0.1" and not ip.startswith("127.")
    return False


def lan_candidates() -> list[str]:
    """All private LAN IPs of this machine (VPN tunnel exits excluded by preference)."""
    found: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if _is_private_lan(ip) and ip not in found:
                found.append(ip)
    except socket.gaierror:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if _is_private_lan(ip) and ip not in found:
                found.append(ip)
    except OSError:
        pass
    return sorted(
        found, key=lambda ip: (0 if ip.startswith("192.168.") else 1 if ip.startswith("172.") else 2, ip)
    )


def lan_token() -> tuple[str, bool]:
    """Stable per-machine demo token: generated once, reused on restarts so
    open tabs and QR codes survive a server restart. Never printed in full
    logs; file mode 600."""
    path = ROOT / "data" / "api_token"
    if path.is_file():
        return path.read_text().strip(), False
    token = secrets.token_urlsafe(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as file:
        file.write(token)
    return token, True


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def resolve_bind(host_arg, lan, api_token_arg, env_token):
    """Pure host/token resolution (no side effects, unit-testable).

    Returns (host, token_or_empty). Refuses a network-visible bind without
    any token source: without API_TOKEN the backend serves everything as
    master (see main.py), so --host 0.0.0.0 without a token would expose an
    open server to the network.
    """
    host = host_arg or ("0.0.0.0" if lan else "127.0.0.1")
    token = api_token_arg or env_token or ""
    if host not in LOOPBACK_HOSTS and not token and not lan:
        raise RuntimeError(
            f"Отказ: сервер открыт в сеть (host={host}), а токена нет. "
            "Укажи --api-token ... или задай API_TOKEN=... в окружении. "
            "Без токена API отдаёт всё без авторизации."
        )
    return host, token


def lan_ip() -> str:
    candidates = lan_candidates()
    if not candidates:
        raise RuntimeError("Не найден LAN IP. Укажи --host и --public-url вручную.")
    if len(candidates) > 1:
        print(f"Найдено несколько LAN IP {candidates}, беру {candidates[0]}. "
              "Другой — через --host/--public-url.", flush=True)
    return candidates[0]


def ensure_cert(ip: str) -> tuple[Path, Path]:
    """Self-signed HTTPS cert for the LAN IP (phone browsers need TLS for camera)."""
    certs = ROOT / "data" / "certs"
    certs.mkdir(parents=True, exist_ok=True)
    crt, key = certs / f"lan-{ip}.crt", certs / f"lan-{ip}.key"
    if not crt.exists() or not key.exists():
        if not shutil.which("openssl"):
            raise RuntimeError("Нужен openssl для --lan (генерация self-signed серта).")
        proc = subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(crt), "-days", "3",
                "-subj", f"/CN={ip}", "-addext", f"subjectAltName=IP:{ip}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"openssl не смог выпустить серт: {proc.stderr.strip()}")
    return crt, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Serve on LAN via HTTPS and show a phone QR in the UI (phone as camera).",
    )
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--public-url", default=None)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--https-cert", default=None)
    parser.add_argument("--https-key", default=None)
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError(
            "Use Python 3.11 or 3.12. Windows: py -3.11 run.py; macOS: python3.11 run.py"
        )
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        print("Creating local Python environment...", flush=True)
        venv.create(env, with_pip=True, symlinks=sys.platform != "win32")
    stamp = env / "safety-setup.sha256"
    fingerprint = hashlib.sha256(
        (ROOT / "backend/pyproject.toml").read_bytes()
        + (ROOT / "backend/constraints.txt").read_bytes()
    ).hexdigest()
    if not stamp.exists() or stamp.read_text() != fingerprint:
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "-e",
                ".[vision,linq]",
                "-c",
                "constraints.txt",
            ],
            ROOT / "backend",
        )
        stamp.write_text(fingerprint)
    dist = ROOT / "frontend/dist/index.html"
    sources = [
        *(ROOT / "frontend/src").rglob("*"),
        ROOT / "frontend/package-lock.json",
        ROOT / "frontend/index.html",
        ROOT / "frontend/vite.config.ts",
    ]
    if not dist.exists() or any(
        p.is_file() and p.stat().st_mtime > dist.stat().st_mtime for p in sources
    ):
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError(
                "Install Node.js 22 LTS from https://nodejs.org, reopen your terminal and run again."
            )
        run([npm, "ci", "--no-audit", "--no-fund"], ROOT / "frontend")
        run([npm, "run", "build"], ROOT / "frontend")
    run([python, "scripts/download_model.py"], ROOT / "backend")
    if args.setup_only:
        print("Setup complete. Run this command again without --setup-only to start.")
        return

    def open_browser(health_url, open_url):
        context = ssl._create_unverified_context()
        for _ in range(60):
            try:
                with urllib.request.urlopen(health_url + "/health", timeout=1, context=context):
                    webbrowser.open(open_url)
                    return
            except OSError:
                time.sleep(0.5)

    host, token = resolve_bind(
        args.host, args.lan, args.api_token, os.environ.get("API_TOKEN", "")
    )
    cert, key = args.https_cert, args.https_key
    if args.lan:
        ip = lan_ip()
        auto_crt, auto_key = ensure_cert(ip)
        cert, key = cert or str(auto_crt), key or str(auto_key)
        base_url = (args.public_url or f"https://{ip}:{args.port}").split("?")[0]
        if not token:
            token, fresh = lan_token()
            if fresh:
                print("Новый LAN-токен создан (data/api_token, только твоя машина).", flush=True)
        os.environ["PUBLIC_URL"] = base_url
        os.environ["EXTRA_ORIGINS"] = base_url
        # Pairing link (with token) goes only to the local operator browser;
        # it is never served back through system/info (see main.py).
        pairing_url = base_url + f"?token={token}"
    else:
        if args.public_url:
            os.environ["PUBLIC_URL"] = args.public_url
        base_url = args.public_url or f"http://127.0.0.1:{args.port}"
        pairing_url = base_url + (f"?token={token}" if token else "")
    if token:
        os.environ["API_TOKEN"] = token

    if not args.no_browser:
        threading.Thread(
            target=open_browser, args=(base_url, pairing_url), daemon=True
        ).start()
    print(f"\nIndustrial Safety AI: {base_url}", flush=True)
    if args.lan:
        print(
            "Режим телефона: pairing-ссылка с одноразовым токеном открыта в браузере; "
            "QR в разделе «Камеры и правила» подхватит токен из адресной строки. "
            "Токен нигде не печатается и не отдаётся через API. "
            "Только доверенная сеть: сервер открыт в LAN по токену.",
            flush=True,
        )
    print("Press Ctrl+C to stop.\n", flush=True)
    command = [
        python,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(args.port),
    ]
    if cert and key:
        command += ["--ssl-certfile", cert, "--ssl-keyfile", key]
    if token:
        # Query strings (with ?token=) land in access logs: disable them while
        # the demo token gate is active. Errors are still logged.
        command += ["--no-access-log"]
    run(command, ROOT / "backend")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"\nSetup/start failed: {exc}", file=sys.stderr)
        sys.exit(1)
