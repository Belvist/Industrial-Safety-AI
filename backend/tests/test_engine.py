import pytest
from pydantic import ValidationError

from app.domain.models import BoundingBox, Detection
from app.services.engine import (
    TemporalEngine,
    Tracker,
    assign_detections,
    cleared_assigned,
    contains,
    evaluate,
)
from app.services.schemas import CameraConfig


def detection(name, box=(0.2, 0.1, 0.7, 0.95), confidence=0.9):
    return Detection(name, confidence, BoundingBox(*box))


def test_one_incident_then_resolved_and_reopened():
    engine = TemporalEngine()
    key = (1, "no_helmet")
    assert engine.update({key: 0.9}, 0, 2, 2) == []
    assert engine.update({key: 0.9}, 1, 2, 2) == []
    event = engine.update({key: 0.9}, 2, 2, 2)
    assert event[0][:2] == ("active", key)
    event[0][2].incident_id = "incident-1"
    assert engine.update({key: 0.9}, 3, 2, 2) == []
    # RESOLVED requires positive evidence (helmet seen), not mere absence.
    assert engine.update({}, 4, 2, 2) == []
    assert engine.update({}, 5, 2, 2, cleared={key: 0.9}) == []
    assert engine.update({}, 6, 2, 2, cleared={key: 0.9}) == []
    assert engine.update({}, 7, 2, 2, cleared={key: 0.9})[0][0] == "resolved"
    assert engine.update({key: 0.9}, 8, 2, 2) == []
    # Re-confirmation after RESOLVED also needs consensus: three positives.
    assert engine.update({key: 0.9}, 9, 2, 2) == []
    assert engine.update({key: 0.9}, 10, 2, 2)[0][0] == "active"


def test_prolonged_unknown_is_lost_not_resolved():
    engine = TemporalEngine()
    key = (1, "no_helmet")
    assert engine.update({key: 0.9}, 0, 2, 2) == []
    assert engine.update({key: 0.9}, 1, 2, 2) == []
    event = engine.update({key: 0.9}, 2, 2, 2)
    assert event[0][0] == "active"
    event[0][2].incident_id = "incident-1"
    assert engine.update({key: 0.9}, 3, 2, 2) == []
    # Unknown for longer than clear_seconds: lost, never resolved.
    assert engine.update({}, 4, 2, 2) == []
    lost = engine.update({}, 5, 2, 2)
    assert lost[0][0] == "lost"
    assert lost[0][2].incident_id == "incident-1"


def test_cleared_evidence_beats_unknown_gap():
    tracks = Tracker().update([detection("person", (0.2, 0.1, 0.5, 0.9))], 0)
    helmet = detection("helmet", (0.28, 0.1, 0.42, 0.25), 0.9)
    assignments, _ = assign_detections(tracks, [detection("person"), helmet], CameraConfig())
    cleared = cleared_assigned(tracks, assignments, CameraConfig())
    assert cleared[(tracks[0].id, "no_helmet")] == 0.9
    # No cigarette anywhere is genuine smoking-clearing evidence (active signal
    # gone), unlike an occluded helmet which stays unknown.
    lonely_assign, _ = assign_detections(tracks, [detection("person")], CameraConfig())
    lonely = cleared_assigned(tracks, lonely_assign, CameraConfig())
    assert lonely == {(tracks[0].id, "smoking"): tracks[0].confidence}
    assert (tracks[0].id, "no_helmet") not in lonely


def test_single_frame_and_missing_frame_reset_confirmation():
    engine = TemporalEngine()
    key = (2, "no_vest")
    engine.update({key: 0.8}, 0, 2, 2)
    engine.update({}, 1, 2, 2)
    assert engine.update({key: 0.8}, 2, 2, 2) == []
    assert engine.update({key: 0.8}, 3, 2, 2) == []
    assert engine.update({key: 0.8}, 4, 2, 2)[0][0] == "active"


def test_gap_does_not_count_as_observed_violation():
    engine = TemporalEngine()
    key = (2, "no_helmet")
    engine.update({key: 0.8}, 0, 2, 2)
    assert engine.update({key: 0.8}, 10, 2, 2) == []
    assert engine.update({key: 0.8}, 11, 2, 2) == []


def test_tracker_keeps_ids_with_reordered_people():
    tracker = Tracker()
    a = detection("person", (0.1, 0.1, 0.3, 0.8))
    b = detection("person", (0.6, 0.1, 0.9, 0.9))
    first = tracker.update([a, b], 0)
    next_frame = tracker.update([b, a], 0.3)
    assert [t.id for t in next_frame] == [first[1].id, first[0].id]
    assert tracker.update([a], 4)[0].id not in {first[0].id, first[1].id}


def test_tracker_second_stage_keeps_id_on_low_iou_jump():
    tracker = Tracker()
    first = tracker.update([detection("person", (0.2, 0.2, 0.4, 0.7))], 0)[0]
    # Shifted by 0.16: IoU ~0.11 < 0.2, but center distance 0.16 < gate 0.25.
    again = tracker.update([detection("person", (0.36, 0.2, 0.56, 0.7))], 0.3)[0]
    assert again.id == first.id


def test_tracker_new_id_on_far_jump():
    tracker = Tracker()
    first = tracker.update([detection("person", (0.2, 0.2, 0.4, 0.7))], 0)[0]
    far = tracker.update([detection("person", (0.55, 0.2, 0.75, 0.7))], 0.3)[0]
    assert far.id != first.id


def test_tracker_survives_single_empty_frame():
    tracker = Tracker()
    first = tracker.update([detection("person", (0.2, 0.2, 0.4, 0.7))], 0)[0]
    assert tracker.update([], 0.5) == []
    again = tracker.update([detection("person", (0.22, 0.2, 0.42, 0.7))], 1.0)[0]
    assert again.id == first.id


def test_unknown_ppe_is_not_missing_and_disabled_rules_do_not_fire():
    tracker = Tracker()
    person = detection("person")
    tracks = tracker.update([person], 0)
    assert evaluate(tracks, [person], CameraConfig()) == {}
    assert tracks[0].ppe == {"helmet": "unknown", "vest": "unknown", "mask": "unknown"}
    missing = detection("no_helmet", (0.3, 0.1, 0.5, 0.25))
    assert evaluate(tracks, [person, missing], CameraConfig(helmet=False)) == {}


def test_mask_and_smoking_rules():
    cfg = CameraConfig(mask=True, smoking=True)
    tracks = Tracker().update([detection("person")], 0)
    no_mask = detection("no_mask", (0.35, 0.2, 0.55, 0.4), 0.88)
    result = evaluate(tracks, [detection("person"), no_mask], cfg)
    assert result == {(tracks[0].id, "no_mask"): 0.88}
    assert tracks[0].ppe["mask"] == "missing"
    # Disabled mask rule stays silent.
    assert (
        evaluate(
            tracks, [detection("person"), no_mask], CameraConfig(mask=False)
        )
        == {}
    )
    # Smoking attaches to the nearest worker.
    tracks = Tracker().update([detection("person")], 0)
    smoke = detection("smoking", (0.4, 0.3, 0.5, 0.4), 0.91)
    result = evaluate(tracks, [detection("person"), smoke], cfg)
    assert result == {(tracks[0].id, "smoking"): 0.91}
    # Far-away cigarette is not assigned to anyone (no false alarm).
    tracks = Tracker().update([detection("person", (0.05, 0.1, 0.2, 0.9))], 0)
    far = detection("smoking", (0.8, 0.8, 0.85, 0.9), 0.9)
    assert evaluate(tracks, [detection("person"), far], cfg) == {}
    # Mask present beats missing on conflict.
    tracks = Tracker().update([detection("person")], 0)
    present = detection("mask", (0.35, 0.2, 0.55, 0.4), 0.95)
    assert (
        evaluate(
            tracks, [detection("person"), no_mask, present], cfg
        )
        == {}
    )
    # Extensions are off by default (MVP scope: helmet/vest/zone).
    defaults = CameraConfig()
    assert defaults.vest is True and defaults.mask is False and defaults.smoking is False
    tracks = Tracker().update([detection("person")], 0)
    assert evaluate(tracks, [detection("person"), no_mask], defaults) == {}


def test_ppe_association_and_conflicting_predictions():
    people = [
        detection("person", (0.1, 0.1, 0.5, 0.95)),
        detection("person", (0.5, 0.1, 0.9, 0.95)),
    ]
    tracks = Tracker().update(people, 0)
    missing = detection("no_helmet", (0.22, 0.1, 0.35, 0.25), 0.9)
    result = evaluate(tracks, [*people, missing], CameraConfig())
    assert result == {(tracks[0].id, "no_helmet"): 0.9}
    positive = detection("helmet", (0.22, 0.1, 0.35, 0.25), 0.95)
    assert evaluate(tracks, [*people, missing, positive], CameraConfig()) == {}


def test_vest_sliver_beside_person_does_not_attach():
    # Narrow vest sliver just inside the x-margin but fully outside the
    # person box: zero overlap => stays unassigned (floor item, not PPE).
    tracks = Tracker().update([detection("person", (0.2, 0.1, 0.5, 0.9))], 0)
    sliver = detection("vest", (0.518, 0.4, 0.54, 0.6), 0.9)
    assert evaluate(tracks, [detection("person"), sliver], CameraConfig()) == {}
    assert tracks[0].ppe["vest"] == "unknown"


def test_vest_without_person_creates_nothing():
    tracks = Tracker().update([], 0)
    assert tracks == []
    assert evaluate(tracks, [detection("vest", (0.3, 0.4, 0.6, 0.7), 0.9)], CameraConfig()) == {}


def test_no_helmet_at_torso_height_is_rejected():
    tracks = Tracker().update([detection("person", (0.2, 0.1, 0.5, 0.9))], 0)
    low = detection("no_helmet", (0.3, 0.55, 0.45, 0.7), 0.9)
    assert evaluate(tracks, [detection("person"), low], CameraConfig()) == {}
    assert tracks[0].ppe["helmet"] == "unknown"


def test_assign_detections_reports_owners():
    from app.services.engine import assign_detections

    tracks = Tracker().update([detection("person", (0.2, 0.1, 0.5, 0.9))], 0)
    helmet = detection("no_helmet", (0.28, 0.1, 0.42, 0.25), 0.9)
    stray = detection("vest", (0.7, 0.7, 0.9, 0.9), 0.9)
    _, owners = assign_detections(tracks, [detection("person"), helmet, stray], CameraConfig())
    assert owners[1] == tracks[0].id
    assert owners[2] is None


def test_weak_person_creates_no_worker_track():
    from dataclasses import replace

    tracker = Tracker()
    weak_person = replace(detection("person", (0.2, 0.1, 0.5, 0.9), 0.3), weak=True)
    assert tracker.update([weak_person], 0) == []
    strong = detection("person", (0.2, 0.1, 0.5, 0.9), 0.9)
    assert len(tracker.update([strong], 0.5)) == 1


def test_weak_detections_never_attach_nor_violate_nor_clear():
    from dataclasses import replace

    from app.services.engine import assign_detections, cleared_assigned

    tracks = Tracker().update([detection("person", (0.2, 0.1, 0.5, 0.9))], 0)
    weak_helmet = replace(
        detection("helmet", (0.28, 0.1, 0.42, 0.25), 0.5), weak=True
    )
    weak_missing = replace(
        detection("no_helmet", (0.28, 0.1, 0.42, 0.25), 0.45), weak=True
    )
    people = [detection("person", (0.2, 0.1, 0.5, 0.9))]
    assignments, owners = assign_detections(
        tracks, [*people, weak_helmet, weak_missing], CameraConfig()
    )
    assert owners[1] is None and owners[2] is None
    assert assignments[tracks[0].id] == {}
    assert evaluate(tracks, [*people, weak_helmet, weak_missing], CameraConfig()) == {}
    assert tracks[0].ppe["helmet"] == "unknown"
    assert cleared_assigned(tracks, assignments, CameraConfig()) == {
        (tracks[0].id, "smoking"): tracks[0].confidence
    }


def test_zone_uses_feet_and_includes_boundary():
    zone = [(0.1, 0.7), (0.8, 0.7), (0.8, 1), (0.1, 1)]
    assert contains(zone, (0.4, 0.7))
    assert not contains(zone, (0.4, 0.2))
    track = Tracker().update([detection("person")], 0)
    observed = evaluate(track, [], CameraConfig(zone=zone, zone_enabled=True))
    assert (track[0].id, "danger_zone") in observed


@pytest.mark.parametrize(
    "zone",
    [
        [(-0.1, 0), (0.5, 0), (0.5, 0.9)],
        [(0.1, 0.1), (0.2, 0.2)],
        [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)],
    ],
)
def test_invalid_zones_rejected(zone):
    with pytest.raises(ValidationError):
        CameraConfig(zone=zone, zone_enabled=True)


def test_single_positive_frame_never_confirms():
    engine = TemporalEngine()
    key = (9, "no_helmet")
    assert engine.update({key: 0.9}, 0, 2, 2) == []
    assert engine.update({}, 1, 2, 2) == []


def test_two_of_four_votes_is_not_enough():
    engine = TemporalEngine()
    key = (9, "no_helmet")
    assert engine.update({key: 0.9}, 0, 2, 2) == []
    assert engine.update({key: 0.9}, 1, 2, 2) == []
    assert engine.update({}, 2, 2, 2) == []
    assert engine.update({}, 3, 2, 2) == []


def test_consensus_survives_single_missed_frame():
    engine = TemporalEngine()
    key = (9, "no_helmet")
    assert engine.update({key: 0.9}, 0, 2, 2) == []
    assert engine.update({}, 1, 2, 2) == []
    assert engine.update({key: 0.9}, 2, 2, 2) == []
    assert engine.update({key: 0.9}, 3, 2, 2) == []
    event = engine.update({key: 0.9}, 4, 2, 2)
    assert event[0][0] == "active"


def test_custom_window_and_required():
    engine = TemporalEngine()
    key = (9, "no_helmet")
    assert engine.update({key: 0.9}, 0, 0.2, 2, window=2, required=2) == []
    assert engine.update({key: 0.9}, 1, 0.2, 2, window=2, required=2)[0][0] == "active"


def test_temporal_params_validated():
    with pytest.raises(ValidationError):
        CameraConfig(temporal_required=5, temporal_window=4)


def test_sequence_single_then_gap_then_three_confirms_at_end():
    # T,F,F,F,T,T,T with 1s frames, confirm 2s: active only at t=6.
    engine = TemporalEngine()
    key = (9, "no_helmet")
    present = [0, 4, 5, 6]
    for now in range(6):
        assert engine.update({key: 0.9} if now in present else {}, now, 2, 2) == []
    assert engine.update({key: 0.9}, 6, 2, 2)[0][0] == "active"


def test_sequence_alternating_never_confirms():
    # T,F,T,F,T: never 3 positives within any rolling 4-frame window + time.
    engine = TemporalEngine()
    key = (9, "no_helmet")
    for now in range(5):
        assert engine.update({key: 0.9} if now % 2 == 0 else {}, now, 2, 2) == []


def test_sequence_two_then_miss_then_one_needs_fresh_time():
    # T,T,F,T: votes reach 3/4 at t=3 but the state restarted at t=3,
    # so confirm_seconds is not satisfied yet.
    engine = TemporalEngine()
    key = (9, "no_helmet")
    assert engine.update({key: 0.9}, 0, 2, 2) == []
    assert engine.update({key: 0.9}, 1, 2, 2) == []
    assert engine.update({}, 2, 2, 2) == []
    assert engine.update({key: 0.9}, 3, 2, 2) == []
