"""Regression: per-image metric helpers must be used for per-image dicts.

training/evaluate.py once called mean_ap() (flat-list API) with the
per-image dict format, which silently mis-scores (zero ground truth per
class). The correct entry point is mean_ap_images().
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'training'))
import metrics  # noqa: E402


CLASSES = ['helmet', 'no_helmet']


def _sample():
    expected = {
        'img1': [('helmet', [0.1, 0.1, 0.3, 0.3]), ('no_helmet', [0.5, 0.5, 0.7, 0.7])],
        'img2': [('helmet', [0.2, 0.2, 0.4, 0.4])],
    }
    predicted = {
        'img1': [('helmet', [0.11, 0.11, 0.31, 0.31], 0.9),
                 ('vest', [0.8, 0.8, 0.9, 0.9], 0.4)],
        'img2': [('helmet', [0.21, 0.21, 0.41, 0.41], 0.8)],
    }
    return expected, predicted


def test_mean_ap_images_accepts_per_image_dicts():
    expected, predicted = _sample()
    result = metrics.mean_ap_images(expected, predicted, CLASSES, [0.5])
    assert set(result) == set(CLASSES)
    assert result['helmet'] > 0.9
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_flat_mean_ap_misreads_per_image_dicts():
    # Guard against reintroducing mean_ap() for dict-format inputs:
    # it iterates dict keys as ground truth and finds none per class.
    expected, predicted = _sample()
    flat_pred = [p for items in predicted.values() for p in items]
    try:
        result = metrics.mean_ap(expected, flat_pred, CLASSES, [0.5])
    except (TypeError, ValueError, ZeroDivisionError):
        return
    assert result['helmet'] == 0.0, 'flat mean_ap must not score dict inputs'
