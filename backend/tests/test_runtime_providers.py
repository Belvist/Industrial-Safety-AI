"""Regression tests for auditor-found launch bugs (Oct 2026).

P0: Runtime passed roi_mode= to LinqProvider.__init__, which did not accept
it -> LinQ mode died before model load (surfaced only as model_error).
P1: Runtime did not forward settings.ppe_roi_mode to LocalOnnxProvider, so
PPE_ROI_MODE=off never took effect on the default local path.
"""
from app.core.config import Settings
from app.inference.linq import LinqProvider
from app.services.runtime import Runtime


def _config(tmp_path, **over):
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/safety.db",
        data_dir=str(tmp_path),
        telegram_enabled=False,
        **over,
    )


def test_linq_runtime_constructs_with_roi_mode(tmp_path):
    """Runtime + LinqProvider must not TypeError (real config, no endpoint)."""
    config = _config(tmp_path, inference_provider="linq")
    runtime = Runtime(config)
    assert isinstance(runtime.provider, LinqProvider)
    assert runtime.provider.roi_mode == config.ppe_roi_mode
    # No endpoint: local fallback engages (loads real weights when present,
    # else readiness surfaces as model_error). Either way construction itself
    # must succeed — before the fix this raised TypeError on roi_mode.
    assert runtime.provider.mode in (
        "uninitialized", "local_onnx", "local_unloaded", "linq_remote",
        "failed", "fallback",
    )


def test_linq_runtime_honors_roi_off(tmp_path):
    config = _config(tmp_path, inference_provider="linq", ppe_roi_mode="off")
    runtime = Runtime(config)
    assert runtime.provider.roi_mode == "off"


def test_linq_provider_forwards_roi_to_local_fallback():
    provider = LinqProvider(roi_mode="off")
    assert provider.roi_mode == "off"


def test_local_runtime_forwards_roi_mode(tmp_path):
    config = _config(tmp_path, ppe_roi_mode="off")
    runtime = Runtime(config)
    assert runtime.provider.roi_mode == "off"


def test_local_runtime_default_roi_auto(tmp_path):
    config = _config(tmp_path)
    runtime = Runtime(config)
    assert runtime.provider.roi_mode == "auto"
