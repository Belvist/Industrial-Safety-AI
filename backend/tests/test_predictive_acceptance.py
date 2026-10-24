"""Rule invariants complement, but never replace, real neural video validation."""

from app.domain.models import BoundingBox
from app.services.engine import Track
from app.services.safety_radar import SafetyRadar


def track(x, y=0.6):
    return Track(1, BoundingBox(x - 0.025, y - 0.2, x + 0.025, y), 0, 0.99)


def radar():
    r = SafetyRadar()
    r.set_zones(
        {
            "far": [(0.7, 0.4), (0.9, 0.4), (0.9, 0.8), (0.7, 0.8)],
            "near": [(0.5, 0.4), (0.6, 0.4), (0.6, 0.8), (0.5, 0.8)],
        }
    )
    return r


def test_prediction_warns_before_proximity_and_selects_nearest_zone():
    r = radar()
    for i in range(4):
        r.update([track(0.2 + i * 0.04)], "c", i * 0.2)
    assert 0.5 - 0.32 > r.zone_threshold_norm
    assert r.get_current_risk_level(1) == "WARNING"
    p = r.predict(1, 0.6)
    assert p["zone_name"] == "near"
    assert abs(p["intersection"][0] - 0.5) < 1e-6


def test_parallel_motion_and_stop_cancel_future_entry():
    r = radar()
    for i in range(5):
        r.update([track(0.2 + i * 0.04, 0.9)], "c", i * 0.2)
    assert not r.get_events()
    r = radar()
    for i in range(4):
        r.update([track(0.2 + i * 0.04)], "c", i * 0.2)
    assert r.get_current_risk_level(1) == "WARNING"
    r.update([track(0.32)], "c", 0.8)
    assert r.predict(1, 0.8) is None
    assert r.get_current_risk_level(1) == "NORMAL"


def test_actual_warning_lead_time_and_reentry_episode():
    r = radar()
    events = []
    for i in range(12):
        events += r.update([track(0.2 + i * 0.04)], "c", i * 0.2)
    critical = next(e for e in events if e.level == "CRITICAL" and e.zone_name == "near")
    warning = next(
        e for e in events if e.level == "WARNING" and e.episode_id == critical.episode_id
    )
    assert critical.lead_time_seconds == round(critical.timestamp - warning.timestamp, 4)
    old = critical.episode_id
    r.update([track(0.4)], "c", 2.4)
    r.update([track(0.38)], "c", 2.6)
    ev = r.update([track(0.55)], "c", 2.8)
    assert any(e.level == "CRITICAL" and e.zone_name == "near" and e.episode_id != old for e in ev)
    assert r.get_current_risk_level(1, 5) == "UNKNOWN"


def test_overlapping_zones_have_independent_critical_events():
    r = SafetyRadar()
    r.set_zones(
        {
            "one": [(0.3, 0.3), (0.8, 0.3), (0.8, 0.9), (0.3, 0.9)],
            "two": [(0.4, 0.4), (0.7, 0.4), (0.7, 0.8), (0.4, 0.8)],
        }
    )
    events = r.update([track(0.5)], "c", 0)
    assert {e.zone_name for e in events} == {"one", "two"}
    assert len({e.episode_id for e in events}) == 2
    assert not r.update([track(0.5)], "c", 0.2)


def test_temporary_unknown_does_not_invent_a_new_entry_episode():
    r = radar()
    first = r.update([track(0.55)], "c", 0)
    episode = next(e.episode_id for e in first if e.level == "CRITICAL")
    r.update([], "c", 1.1)
    assert r.get_current_risk_level(1, 1.1) == "UNKNOWN"
    again = r.update([track(0.55)], "c", 1.2)
    assert next(e.episode_id for e in again if e.level == "CRITICAL") == episode
