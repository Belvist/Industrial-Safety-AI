"""Metrics contract on a tiny synthetic set with hand-computed values.

GT: cat A [0,0,10,10], dog B [20,20,30,30], cat C [50,50,60,60].
Preds: cat==A @0.9 (TP), cat==A @0.5 (FP dup), dog~B @0.8 (TP, IoU 0.68).
Expected: cat P=0.5 R=0.5 F1=0.5 AP=0.5; dog P=R=F1=AP=1.0; mAP50=0.75.
"""

import importlib.util
from pathlib import Path

import pytest

METRICS = Path(__file__).resolve().parents[2] / "training" / "metrics.py"


def load_metrics():
    spec = importlib.util.spec_from_file_location("ppe_metrics", METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def metrics():
    return load_metrics()


@pytest.fixture(scope="module")
def scene():
    expected = [
        ("cat", [0, 0, 10, 10]),
        ("dog", [20, 20, 30, 30]),
        ("cat", [50, 50, 60, 60]),
    ]
    predicted = [
        ("cat", [0, 0, 10, 10], 0.9),
        ("cat", [0, 0, 10, 10], 0.5),
        ("dog", [21, 21, 31, 31], 0.8),
    ]
    return expected, predicted


def test_match_counts(metrics, scene):
    expected, predicted = scene
    tp, fp, fn, ranked = metrics.match(expected, predicted, 0.5)
    assert (tp.get("cat"), fp.get("cat"), fn.get("cat")) == (1, 1, 1)
    assert (tp.get("dog"), fp.get("dog", 0), fn.get("dog", 0)) == (1, 0, 0)
    assert [r[2] for r in ranked if r[0] == "cat"] == [True, False]


def test_precision_recall_f1(metrics):
    assert metrics.precision_recall_f1(1, 1, 1) == pytest.approx((0.5, 0.5, 0.5))
    assert metrics.precision_recall_f1(1, 0, 0) == pytest.approx((1.0, 1.0, 1.0))
    assert metrics.precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_average_precision_and_map50(metrics, scene):
    expected, predicted = scene
    _, _, _, ranked = metrics.match(expected, predicted, 0.5)
    cat = [r for r in ranked if r[0] == "cat"]
    dog = [r for r in ranked if r[0] == "dog"]
    assert metrics.average_precision(cat, 2) == pytest.approx(0.5)
    assert metrics.average_precision(dog, 1) == pytest.approx(1.0)
    assert metrics.average_precision([], 0) == 0.0
    result = metrics.mean_ap(expected, predicted, ["cat", "dog"])
    assert result == pytest.approx({"cat": 0.5, "dog": 1.0})


def test_confusion_matrix(metrics, scene):
    expected, predicted = scene
    matrix = metrics.confusion_matrix(expected, predicted, ["cat", "dog"], 0.5)
    assert matrix[("cat", "cat")] == 1
    assert matrix[("dog", "dog")] == 1
    assert matrix[("cat", "background")] == 1
    assert matrix[("background", "cat")] == 1
    assert ("dog", "background") not in matrix


def _synthetic_dataset(root, clips=5, bad_label=None):
    from pathlib import Path

    import cv2
    import numpy as np

    root = Path(root)
    pixels = np.zeros((16, 16, 3), np.uint8)
    for c in range(clips):
        images = root / f"clip-{c:02d}" / "images"
        labels = root / f"clip-{c:02d}" / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for f in range(2):
            cv2.imwrite(str(images / f"f{f}.jpg"), pixels)
            content = bad_label if bad_label and c == 0 and f == 0 else "0 0.5 0.5 0.2 0.4\n"
            (labels / f"f{f}.txt").write_text(content)
    return root


def _prepare(*args):
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [sys.executable, str(repo / "training" / "prepare_dataset.py"), *args],
        capture_output=True,
        text=True,
    )


def test_prepare_valid_dataset_and_yaml(tmp_path):
    root = _synthetic_dataset(tmp_path / "ds", clips=5)
    out = tmp_path / "data.yaml"
    result = _prepare(str(root), "--out", str(out))
    assert result.returncode == 0, result.stderr
    text = out.read_text()
    assert "nc: 7" in text
    assert "YAML validated" in result.stdout


def test_prepare_rejects_bad_labels_and_small_sets(tmp_path):
    root = _synthetic_dataset(tmp_path / "bad", clips=5, bad_label="9 0.5 0.5 0.2 0.4\n")
    bad = _prepare(str(root), "--out", str(tmp_path / "x.yaml"))
    assert bad.returncode != 0 and "out of range" in (bad.stdout + bad.stderr)
    small = _synthetic_dataset(tmp_path / "small", clips=2)
    refused = _prepare(str(small), "--out", str(tmp_path / "y.yaml"))
    assert refused.returncode != 0
    assert "independent" in (refused.stdout + refused.stderr)


def test_ap_counts_both_recall_increases_after_false_positive(metrics):
    ranked = [("helmet", 0.99, True), ("helmet", 0.85, False), ("helmet", 0.65, True)]
    assert metrics.average_precision(ranked, 2) == pytest.approx(5 / 6)


def test_same_coordinates_on_different_images_never_match(metrics):
    expected = {"frame_a": [("no_helmet", [0.1, 0.1, 0.3, 0.3])]}
    predicted = {"frame_b": [("no_helmet", [0.1, 0.1, 0.3, 0.3], 0.95)]}
    tp, fp, fn, _ = metrics.match_images(expected, predicted, 0.5)
    assert tp.get("no_helmet", 0) == 0
    assert fp["no_helmet"] == 1
    assert fn["no_helmet"] == 1
    assert metrics.mean_ap_images(expected, predicted, ["no_helmet"])["no_helmet"] == 0.0


def test_ap_aggregates_globally_after_per_frame_matching(metrics):
    gt = {"frame_a": [("helmet", [0, 0, 1, 1])], "frame_b": [("helmet", [0, 0, 1, 1])]}
    pred = {"frame_a": [("helmet", [0, 0, 1, 1], 0.9),
                         ("helmet", [0, 0, 1, 1], 0.8)],
            "frame_b": [("helmet", [0, 0, 1, 1], 0.7)]}
    tp, fp, fn, ranked = metrics.match_images(gt, pred)
    assert tp["helmet"] == 2 and fp["helmet"] == 1 and fn.get("helmet", 0) == 0
    assert [x[1] for x in ranked] == [0.9, 0.8, 0.7]
    assert metrics.mean_ap_images(gt, pred, ["helmet"])["helmet"] == pytest.approx(5 / 6)
