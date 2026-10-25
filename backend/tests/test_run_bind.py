"""Regression tests for the --host/--api-token bind hole (Oct 2026).

P0: run.py applied --api-token only inside --lan. `--host 0.0.0.0
--api-token ...` without --lan silently dropped the CLI token; worse, a bare
--host 0.0.0.0 served the API to the network with an empty token (main.py
treats empty api_token as open local/master mode).
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("run_launcher", ROOT / "run.py")
run_launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_launcher)
resolve_bind = run_launcher.resolve_bind


def test_network_bind_without_token_refused():
    with pytest.raises(RuntimeError):
        resolve_bind("0.0.0.0", False, "", "")


def test_network_bind_with_cli_token_allowed():
    host, token = resolve_bind("0.0.0.0", False, "tok123", "")
    assert (host, token) == ("0.0.0.0", "tok123")


def test_env_token_applies_without_lan():
    host, token = resolve_bind("192.168.1.5", False, "", "envtok")
    assert (host, token) == ("192.168.1.5", "envtok")


def test_cli_token_beats_env_token():
    _, token = resolve_bind("0.0.0.0", False, "cli", "env")
    assert token == "cli"


def test_loopback_without_token_stays_local():
    host, token = resolve_bind(None, False, "", "")
    assert (host, token) == ("127.0.0.1", "")


def test_lan_defaults_to_all_interfaces_pending_generation():
    host, token = resolve_bind(None, True, "", "")
    assert host == "0.0.0.0" and token == ""
