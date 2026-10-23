import numpy as np
import pytest

from app.inference.local_onnx import LocalOnnxProvider


def test_decode_undoes_letterbox_and_per_class_nms():
    pytest.importorskip("cv2")
    provider = LocalOnnxProvider("unused", confidence=0.4)
    provider.classes = {0: "person", 1: "helmet"}
    # A 1280x720 frame is scaled to 640x360 and padded 140px top/bottom.
    output = np.array(
        [
            [
                [320, 320, 100],
                [320, 320, 200],
                [320, 320, 30],
                [360, 360, 30],
                [0.9, 0.8, 0.1],
                [0.1, 0.1, 0.95],
            ]
        ],
        dtype=np.float32,
    )
    result = provider.decode(output, 1280, 720, 0.5, 0, 140)
    assert len(result) == 2
    person = next(d for d in result if d.class_name == "person")
    assert person.bbox.x1 == pytest.approx(0.25)
    assert person.bbox.x2 == pytest.approx(0.75)
    assert person.bbox.y1 == pytest.approx(0)
    assert person.bbox.y2 == pytest.approx(1)


def _no_helmet_output():
    # Two no_helmet candidates, far apart: strong 0.90 and weak 0.45.
    return np.array(
        [
            [
                [200, 500],
                [200, 500],
                [80, 80],
                [80, 80],
                [0.05, 0.10],
                [0.90, 0.45],
            ]
        ],
        dtype=np.float32,
    )


def test_weak_no_helmet_filtered_by_per_class_threshold():
    pytest.importorskip("cv2")
    provider = LocalOnnxProvider(
        "unused", confidence=0.4, thresholds={"no_helmet": 0.60}
    )
    provider.classes = {0: "person", 1: "no_helmet"}
    result = provider.decode(_no_helmet_output(), 640, 640, 1.0, 0, 0)
    assert [(d.class_name, d.weak) for d in result] == [
        ("no_helmet", False),
        ("no_helmet", True),
    ]
    assert result[0].confidence == pytest.approx(0.90)


def test_sub_threshold_kept_weak_and_below_floor_dropped():
    pytest.importorskip("cv2")
    provider = LocalOnnxProvider(
        "unused", confidence=0.4, thresholds={"no_helmet": 0.60}, floor=0.25
    )
    provider.classes = {0: "person", 1: "no_helmet"}
    output = np.array(
        [
            [
                [200, 500, 100],
                [200, 500, 100],
                [80, 80, 80],
                [80, 80, 80],
                [0.05, 0.10, 0.05],
                [0.90, 0.45, 0.10],
            ]
        ],
        dtype=np.float32,
    )
    result = provider.decode(output, 640, 640, 1.0, 0, 0)
    by_conf = {round(d.confidence, 2): d.weak for d in result}
    assert by_conf == {0.90: False, 0.45: True}
    # 0.10 is below the display floor and disappears entirely.


def test_same_weak_detection_passes_with_low_global_threshold():
    pytest.importorskip("cv2")
    provider = LocalOnnxProvider("unused", confidence=0.4)
    provider.classes = {0: "person", 1: "no_helmet"}
    result = provider.decode(_no_helmet_output(), 640, 640, 1.0, 0, 0)
    assert len(result) == 2
    assert not any(d.weak for d in result)


def test_threshold_for_falls_back_to_global():
    provider = LocalOnnxProvider("unused", confidence=0.4, thresholds={"helmet": 0.7})
    assert provider.threshold_for("helmet") == 0.7
    assert provider.threshold_for("vest") == 0.4


def test_supported_classes_reflects_model_heads():
    provider = LocalOnnxProvider("unused")
    provider.classes = {0: "person", 1: "helmet", 2: "smoking", 3: "ladder"}
    assert provider.supported_classes() == ["helmet", "person", "smoking"]
    assert "smoking" not in LocalOnnxProvider("unused").supported_classes()


def _stub_person(a, b, c, d):
    from app.domain.models import BoundingBox, Detection
    return Detection("person", 0.93, BoundingBox(a, b, c, d))


def _stub_ppe(name, confidence=0.7, weak=False):
    from app.domain.models import BoundingBox, Detection
    return Detection(name, confidence, BoundingBox(0.4, 0.1, 0.6, 0.25), weak=weak)


class _PersonStub:
    def __init__(self, people):
        self.people = people

    def predict_sync(self, frame):
        return self.people


def test_roi_projects_to_original_frame_and_preserves_weak(monkeypatch):
    from app.inference.local_onnx import person_roi_pixels, project_roi_detection
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    person = _stub_person(0.25, 0.2, 0.75, 0.9)
    roi = person_roi_pixels(person.bbox, 640, 480)
    x1, y1, x2, y2 = roi
    projected = project_roi_detection(_stub_ppe("no_helmet", 0.44, True), roi, 640, 480)
    assert projected.bbox.x1 == pytest.approx((x1 + 0.4 * (x2 - x1)) / 640)
    assert projected.bbox.y2 == pytest.approx((y1 + 0.25 * (y2 - y1)) / 480)
    assert projected.weak is True

    provider = LocalOnnxProvider("unused", roi_mode="auto")
    provider.person_detector = _PersonStub([person])
    sizes = []
    def fake_infer(crop):
        sizes.append(crop.shape)
        return [_stub_ppe("no_helmet", 0.44, True), _stub_ppe("person")]
    monkeypatch.setattr(provider, "_predict_single", fake_infer)
    result = provider.predict_sync(frame)
    assert len(sizes) == 1 and sizes[0][1] < 640
    assert [d.class_name for d in result] == ["no_helmet", "person"]
    assert result[0].weak and result[0].bbox.x1 == projected.bbox.x1
    assert provider.status()["ppe_roi_mode"] == "auto"


def test_adaptive_roi_falls_back_for_three_people(monkeypatch):
    provider = LocalOnnxProvider("unused", roi_mode="auto")
    provider.person_detector = _PersonStub([
        _stub_person(.01,.1,.25,.8), _stub_person(.3,.1,.55,.8),
        _stub_person(.6,.1,.92,.8),
    ])
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    images = []
    def fake_infer(crop):
        images.append(crop.shape)
        return [_stub_ppe("no_vest", 0.42, True)]
    monkeypatch.setattr(provider, "_predict_single", fake_infer)
    result = provider.predict_sync(frame)
    assert images == [frame.shape] and result[0].weak


def test_roi_is_optional_and_deduplicates_by_class(monkeypatch):
    from app.inference.local_onnx import deduplicate_ppe
    provider = LocalOnnxProvider("unused", roi_mode="off")
    provider.person_detector = _PersonStub([_stub_person(.25,.2,.75,.9)])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    images = []
    def fake_infer(crop):
        images.append(crop.shape)
        return [_stub_ppe("helmet", 0.7)]
    monkeypatch.setattr(provider, "_predict_single", fake_infer)
    assert [d.class_name for d in provider.predict_sync(frame)] == ["helmet", "person"]
    assert images == [frame.shape]
    merged = deduplicate_ppe([
        _stub_ppe("helmet", .9), _stub_ppe("helmet", .65),
        _stub_ppe("no_helmet", .61),
    ])
    assert [(d.class_name, d.confidence) for d in merged] == [
        ("helmet", .9), ("no_helmet", .61),
    ]
