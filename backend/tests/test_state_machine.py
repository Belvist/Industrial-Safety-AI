import math

import pytest

from app.domain.models import BoundingBox, Detection
from app.services.engine import TemporalEngine, ViolationPhase
from app.services.schemas import CameraConfig
from app.services.state_machine import CameraEvent, CameraMachine, CameraState


def test_camera_normal_lifecycle_and_recovery():
    machine = CameraMachine()
    for event, state in [
        (CameraEvent.START, CameraState.STARTING),
        (CameraEvent.READY, CameraState.RUNNING),
        (CameraEvent.FRAME, CameraState.RUNNING),
        (CameraEvent.STALE, CameraState.STALE),
        (CameraEvent.FRAME, CameraState.RUNNING),
        (CameraEvent.STOP, CameraState.STOPPING),
        (CameraEvent.STOPPED, CameraState.STOPPED),
    ]:
        assert machine.dispatch(event) == state
    assert machine.version == 7
    machine.dispatch(CameraEvent.START)
    machine.dispatch(CameraEvent.FAIL)
    assert machine.state == CameraState.ERROR
    machine.dispatch(CameraEvent.START)
    machine.dispatch(CameraEvent.READY)
    machine.dispatch(CameraEvent.EOF)
    assert machine.state == CameraState.ENDED


def test_camera_rejects_late_frame_and_double_start():
    machine = CameraMachine()
    with pytest.raises(RuntimeError):
        machine.dispatch(CameraEvent.FRAME)
    machine.dispatch(CameraEvent.START)
    with pytest.raises(RuntimeError):
        machine.dispatch(CameraEvent.START)
    machine.dispatch(CameraEvent.STOP)
    with pytest.raises(RuntimeError):
        machine.dispatch(CameraEvent.READY)
    machine.dispatch(CameraEvent.STOPPED)
    assert machine.state == CameraState.STOPPED


def test_upload_reservation_excludes_camera_start():
    machine = CameraMachine()
    machine.dispatch(CameraEvent.UPLOAD)
    with pytest.raises(RuntimeError):
        machine.dispatch(CameraEvent.START)
    machine.dispatch(CameraEvent.STOP)
    machine.dispatch(CameraEvent.STOPPED)
    assert machine.state == CameraState.STOPPED


def test_active_incident_clearing_cancels_without_duplicate():
    engine = TemporalEngine()
    key = (1, "no_helmet")
    engine.update({key: 0.9}, 0, 2, 2)
    engine.update({key: 0.9}, 1, 2, 2)
    events = engine.update({key: 0.9}, 2, 2, 2)
    events[0][2].incident_id = "one"
    assert engine.states[key].phase == ViolationPhase.ACTIVE
    assert engine.update({}, 3, 2, 2) == []
    assert engine.states[key].phase == ViolationPhase.CLEARING
    assert engine.update({key: 0.91}, 3.5, 2, 2) == []
    assert engine.states[key].phase == ViolationPhase.ACTIVE
    assert engine.states[key].incident_id == "one"


def test_gap_interrupts_confirmed_incident_without_false_resolution():
    engine = TemporalEngine()
    key = (1, "danger_zone")
    engine.update({key: 0.9}, 0, 2, 2)
    engine.update({key: 0.9}, 1, 2, 2)
    events = engine.update({key: 0.9}, 2, 2, 2)
    events[0][2].incident_id = "one"
    events = engine.update({key: 0.9}, 10, 2, 2)
    assert [e[0] for e in events] == ["interrupted"]
    assert engine.states[key].phase == ViolationPhase.PENDING
    assert engine.states[key].incident_id is None
    with pytest.raises(ValueError):
        engine.update({}, 9, 2, 2)


@pytest.mark.parametrize(
    "box", [(0, 0, 0, 1), (-0.1, 0, 0.5, 1), (0, 0, math.nan, 1), (0, 0.5, 1, 0.4)]
)
def test_invalid_model_boxes_rejected(box):
    with pytest.raises(ValueError):
        BoundingBox(*box)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.nan])
def test_invalid_model_confidence_rejected(confidence):
    with pytest.raises(ValueError):
        Detection("person", confidence, BoundingBox(0, 0, 1, 1))


def test_self_intersecting_zone_rejected_even_if_signed_area_not_zero():
    with pytest.raises(ValueError):
        CameraConfig(zone=[(0.1, 0.1), (0.9, 0.9), (0.1, 0.9), (0.6, 0.1)], zone_enabled=True)
