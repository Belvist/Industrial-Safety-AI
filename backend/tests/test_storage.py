"""Incident terminal transitions: single finish_incident path for all outcomes."""

import pytest

from app.services.storage import Storage


def _storage(tmp_path):
    return Storage(tmp_path / "s.db")


def _active(storage):
    return storage.create_incident("cam", "C", (1, "no_helmet"), 0.9, 1.0)


@pytest.mark.parametrize("reason", ["frame_gap", "observation_lost", "monitoring_stopped"])
def test_all_interrupt_reasons_finish_with_end_time(tmp_path, reason):
    storage = _storage(tmp_path)
    record = _active(storage)
    done = storage.finish_incident(record["id"], "interrupted", reason, duration=4.0)
    assert done["status"] == "interrupted"
    assert done["resolution_reason"] == reason
    assert done["ended_at"] is not None
    assert done["duration"] == 4.0


def test_resolved_finish_and_double_finish_rejected(tmp_path):
    storage = _storage(tmp_path)
    record = _active(storage)
    done = storage.finish_incident(record["id"], "resolved", "violation_cleared")
    assert done["status"] == "resolved"
    assert done["ended_at"] == done["resolved_at"]
    with pytest.raises(ValueError):
        storage.finish_incident(record["id"], "resolved", "violation_cleared")
    with pytest.raises(ValueError):
        storage.finish_incident(record["id"], "interrupted", "frame_gap")
    assert storage.get(record["id"])["status"] == "resolved"
    with pytest.raises(ValueError):
        storage.finish_incident("missing", "resolved", "violation_cleared")
    with pytest.raises(ValueError):
        storage.finish_incident(record["id"], "active", "violation_cleared")



def test_interrupted_does_not_claim_resolved_at(tmp_path):
    storage = _storage(tmp_path)
    incident = _active(storage)
    done = storage.finish_incident(incident["id"], "interrupted", "source_lost")
    assert done["status"] == "interrupted"
    assert done["ended_at"] is not None
    assert done["resolved_at"] is None
