from app.core.config import ROOT, Settings


def test_env_example_class_thresholds_match_settings_defaults():
    env_values = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env_values[key] = value

    settings = Settings(_env_file=None)
    assert float(env_values["HELMET_THRESHOLD"]) == settings.helmet_threshold
    assert float(env_values["NO_HELMET_THRESHOLD"]) == settings.no_helmet_threshold
    assert float(env_values["VEST_THRESHOLD"]) == settings.vest_threshold
    assert float(env_values["NO_VEST_THRESHOLD"]) == settings.no_vest_threshold
