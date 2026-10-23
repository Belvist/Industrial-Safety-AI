"""Pure detection metrics (no heavy deps): TP/FP/FN matching, P/R/F1, AP, mAP.

Single source of truth used by training/evaluate.py and backend tests.
All boxes are [x1, y1, x2, y2] in identical coordinates.
"""

from __future__ import annotations


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def match(
    expected: list[tuple[str, list[float]]],
    predicted: list[tuple[str, list[float], float]],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], list[tuple[str, float, bool]]]:
    """Greedy one-to-one matching (highest confidence first).

    Returns (tp, fp, fn) per class plus ranked (class, conf, is_tp) list for AP.
    """
    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    ranked: list[tuple[str, float, bool]] = []
    taken: set[int] = set()
    for cls, box, conf in sorted(predicted, key=lambda p: -p[2]):
        hit = None
        best = iou_threshold
        for i, (gt_cls, gt_box) in enumerate(expected):
            if i in taken or gt_cls != cls:
                continue
            value = iou(box, gt_box)
            if value >= best:
                best = value
                hit = i
        if hit is None:
            fp[cls] = fp.get(cls, 0) + 1
            ranked.append((cls, conf, False))
        else:
            taken.add(hit)
            tp[cls] = tp.get(cls, 0) + 1
            ranked.append((cls, conf, True))
    for i, (gt_cls, _) in enumerate(expected):
        if i not in taken:
            fn[gt_cls] = fn.get(gt_cls, 0) + 1
    return tp, fp, fn, ranked


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def average_precision(ranked: list[tuple[str, float, bool]], total_gt: int) -> float:
    """All-point interpolated AP on confidence-ranked detections.

    Precision is enveloped from high to low recall, then integrated only
    over *increases* in recall. False positives never increase recall.
    """
    if total_gt <= 0:
        return 0.0
    ranked = sorted(ranked, key=lambda row: -row[1])
    recalls = [0.0]
    precisions = [1.0]
    hits = 0
    for i, (_, _, is_tp) in enumerate(ranked, 1):
        if is_tp:
            hits += 1
        recalls.append(hits / total_gt)
        precisions.append(hits / i)
    recalls.append(1.0)
    precisions.append(0.0)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    return sum(
        (recalls[i] - recalls[i - 1]) * precisions[i]
        for i in range(1, len(recalls))
        if recalls[i] > recalls[i - 1]
    )


def mean_ap(
    expected: list[tuple[str, list[float]]],
    predicted: list[tuple[str, list[float], float]],
    classes: list[str],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    """mAP over classes; pass 0.5..0.95 for mAP50-95."""
    if iou_thresholds is None:
        iou_thresholds = [0.5]
    per_class: dict[str, list[float]] = {c: [] for c in classes}
    for threshold in iou_thresholds:
        _, _, _, ranked = match(expected, predicted, threshold)
        for cls in classes:
            total = sum(1 for gt_cls, _ in expected if gt_cls == cls)
            per_class[cls].append(
                average_precision([r for r in ranked if r[0] == cls], total)
            )
    return {cls: sum(v) / len(v) for cls, v in per_class.items()}


def confusion_matrix(
    expected: list[tuple[str, list[float]]],
    predicted: list[tuple[str, list[float], float]],
    classes: list[str],
    iou_threshold: float = 0.5,
) -> dict[tuple[str, str], int]:
    """(gt_class, pred_class) counts; 'background' for misses/false alarms."""
    matrix: dict[tuple[str, str], int] = {}
    taken: set[int] = set()
    for cls, box, _ in sorted(predicted, key=lambda p: -p[2]):
        hit = None
        best = iou_threshold
        for i, (gt_cls, gt_box) in enumerate(expected):
            if i in taken or gt_cls != cls:
                continue
            value = iou(box, gt_box)
            if value >= best:
                best = value
                hit = i
        if hit is None:
            matrix[("background", cls)] = matrix.get(("background", cls), 0) + 1
        else:
            taken.add(hit)
            matrix[(expected[hit][0], cls)] = matrix.get((expected[hit][0], cls), 0) + 1
    for i, (gt_cls, _) in enumerate(expected):
        if i not in taken:
            matrix[(gt_cls, "background")] = matrix.get((gt_cls, "background"), 0) + 1
    return matrix


def match_images(
    expected_by_image: dict[str, list[tuple[str, list[float]]]],
    predicted_by_image: dict[str, list[tuple[str, list[float], float]]],
    iou_threshold: float = 0.5,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], list[tuple[str, float, bool]]]:
    """Match detections *within each frame*. Boxes from different images
    must never match, even if their normalized coordinates are identical.
    AP ranking is aggregated globally only AFTER per-image TP/FP labels.
    """
    total_tp: dict[str, int] = {}
    total_fp: dict[str, int] = {}
    total_fn: dict[str, int] = {}
    ranked: list[tuple[str, float, bool]] = []
    for image_id in set(expected_by_image) | set(predicted_by_image):
        tp, fp, fn, scores = match(
            expected_by_image.get(image_id, []),
            predicted_by_image.get(image_id, []),
            iou_threshold,
        )
        for source, dest in ((tp, total_tp), (fp, total_fp), (fn, total_fn)):
            for cls, number in source.items():
                dest[cls] = dest.get(cls, 0) + number
        ranked.extend(scores)
    ranked.sort(key=lambda row: -row[1])
    return total_tp, total_fp, total_fn, ranked


def mean_ap_images(
    expected_by_image: dict[str, list[tuple[str, list[float]]]],
    predicted_by_image: dict[str, list[tuple[str, list[float], float]]],
    classes: list[str],
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    thresholds = iou_thresholds if iou_thresholds is not None else [0.5]
    if not thresholds:
        raise ValueError("At least one IoU threshold is required")
    result = {cls: [] for cls in classes}
    total_gt = {
        cls: sum(1 for items in expected_by_image.values() for label, _ in items if label == cls)
        for cls in classes
    }
    for threshold in thresholds:
        _, _, _, ranked = match_images(expected_by_image, predicted_by_image, threshold)
        for cls in classes:
            result[cls].append(
                average_precision([row for row in ranked if row[0] == cls], total_gt[cls])
            )
    return {cls: sum(values) / len(values) for cls, values in result.items()}


def confusion_matrix_images(
    expected_by_image: dict[str, list[tuple[str, list[float]]]],
    predicted_by_image: dict[str, list[tuple[str, list[float], float]]],
    classes: list[str],
    iou_threshold: float = 0.5,
) -> dict[tuple[str, str], int]:
    matrix: dict[tuple[str, str], int] = {}
    for image_id in set(expected_by_image) | set(predicted_by_image):
        local = confusion_matrix(
            expected_by_image.get(image_id, []),
            predicted_by_image.get(image_id, []),
            classes,
            iou_threshold,
        )
        for key, count in local.items():
            matrix[key] = matrix.get(key, 0) + count
    return matrix
