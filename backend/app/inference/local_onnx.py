"""YOLOv8 ONNX adapter. Input is uint8 BGR; output boxes are normalized xyxy."""

import ast
import asyncio
from math import ceil, floor
from pathlib import Path
from threading import Lock

import numpy as np

from app.domain.models import BoundingBox, Detection
from app.inference.base import Frame, InferenceProvider

ALIASES = {
    "person": "person",
    "hardhat": "helmet",
    "helmet": "helmet",
    "no-hardhat": "no_helmet",
    "no-helmet": "no_helmet",
    "safety vest": "vest",
    "vest": "vest",
    "no-safety vest": "no_vest",
    "no-vest": "no_vest",
    # Respirator / face mask family (hackathon case: "снял респиратор").
    "mask": "mask",
    "face-mask": "mask",
    "face_mask": "mask",
    "respirator": "mask",
    "no-mask": "no_mask",
    "no_mask": "no_mask",
    "no-respirator": "no_mask",
    "without-mask": "no_mask",
    "without mask": "no_mask",
    # Smoking family (hackathon case: "закурил рядом с горючими материалами").
    "cigarette": "smoking",
    "cigar": "smoking",
    "smoke": "smoking",
    "smoking": "smoking",
    "smoking-person": "smoking",
}

# Classes the tracker/rules understand. PPE weights may not contain the mask or
# smoking heads yet (they arrive with the LinQ model) — decode() passes through
# whatever the weights provide instead of failing, so missing heads stay silent.
KNOWN_CLASSES = {
    "person",
    "helmet",
    "no_helmet",
    "vest",
    "no_vest",
    "mask",
    "no_mask",
    "smoking",
}



def person_roi_pixels(b: BoundingBox, w: int, h: int) -> tuple[int, int, int, int]:
    """Pad around a worker, including any helmet above the detected head."""
    bw, bh = b.x2 - b.x1, b.y2 - b.y1
    return (
        max(0, floor((b.x1 - 0.12 * bw) * w)),
        max(0, floor((b.y1 - 0.13 * bh) * h)),
        min(w, ceil((b.x2 + 0.12 * bw) * w)),
        min(h, ceil((b.y2 + 0.05 * bh) * h)),
    )


def project_roi_detection(d: Detection, roi: tuple[int, int, int, int], w: int, h: int) -> Detection:
    """Project crop-normalized coordinates into full-frame normalized coordinates."""
    x1, y1, x2, y2 = roi
    b = d.bbox
    return Detection(
        d.class_name, d.confidence,
        BoundingBox(
            (x1 + b.x1 * (x2 - x1)) / w, (y1 + b.y1 * (y2 - y1)) / h,
            (x1 + b.x2 * (x2 - x1)) / w, (y1 + b.y2 * (y2 - y1)) / h,
        ), track_hint=d.track_hint, weak=d.weak,
    )


def deduplicate_ppe(detections: list[Detection], threshold: float = 0.55) -> list[Detection]:
    """Remove duplicate detections across overlapping worker crops, by class."""
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda item: -item.confidence):
        b = d.bbox
        duplicate = False
        for other in kept:
            if d.class_name != other.class_name:
                continue
            o = other.bbox
            intersection = (
                max(0.0, min(b.x2, o.x2) - max(b.x1, o.x1))
                * max(0.0, min(b.y2, o.y2) - max(b.y1, o.y1))
            )
            a1 = (b.x2 - b.x1) * (b.y2 - b.y1)
            a2 = (o.x2 - o.x1) * (o.y2 - o.y1)
            union = a1 + a2 - intersection
            if union > 0 and intersection / union >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


class LocalOnnxProvider(InferenceProvider):
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.4,
        threads: int = 2,
        person_model_path: str | None = None,
        required_classes: set[str] | None = None,
        providers: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
        floor: float = 0.25,
        roi_mode: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.threads = threads
        self._session = None
        self._lock = Lock()
        self.classes: dict[int, str] = {}
        # Per-class detection thresholds (sprint accuracy §1). Absent entries
        # fall back to the global confidence so old call sites keep working.
        self.thresholds = dict(thresholds) if thresholds else {}
        # Display floor (fix-visibility F2): detections below it are dropped;
        # between floor and class threshold they are kept flagged weak.
        self.floor = floor
        if roi_mode not in {"auto", "off"}:
            raise ValueError("roi_mode must be auto or off")
        self.roi_mode = roi_mode
        # Requested execution providers, in priority order. CPU is always
        # appended as the last fallback; the FACT of what runs is read back
        # from session.get_providers() into active_providers.
        self.requested_providers = list(providers) if providers else []
        self.active_providers: list[str] = []
        self.required_classes = required_classes or {
            "person",
            "helmet",
            "no_helmet",
            "vest",
            "no_vest",
        }
        self.person_detector = (
            LocalOnnxProvider(
                person_model_path,
                confidence,
                threads,
                required_classes={"person"},
                providers=self.requested_providers,
                thresholds=self.thresholds,
                floor=self.floor,
                roi_mode="off",
            )
            if person_model_path
            else None
        )

    def threshold_for(self, class_name: str) -> float:
        return self.thresholds.get(class_name, self.confidence)

    def supported_classes(self) -> list[str]:
        """Detector classes this model actually emits (subset of KNOWN_CLASSES).

        A rule whose class is absent (e.g. smoking with the current PPE
        weights) can never fire — the UI must say so honestly instead of
        pretending the violation is monitored.
        """
        return sorted(set(self.classes.values()) & KNOWN_CLASSES)

    @property
    def name(self) -> str:
        return "local_onnx"

    def status(self) -> dict:
        return {
            "provider": "local_onnx",
            "mode": "local_onnx",
            "requested_providers": self.requested_providers,
            "active_providers": list(self.active_providers),
            "supported_classes": self.supported_classes(),
            "model": Path(self.model_path).name,
            "ppe_roi_mode": self.roi_mode if self.person_detector else "off",
        }

    def load(self) -> None:
        if self.person_detector:
            self.person_detector.load()
        with self._lock:
            if self._session is not None:
                return
            if not self.model_path.is_file():
                raise RuntimeError("Модель не найдена. Запустите python scripts/download_model.py.")
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            available = set(ort.get_available_providers())
            chosen = [p for p in self.requested_providers if p in available]
            if "CPUExecutionProvider" not in chosen:
                if "CPUExecutionProvider" not in available:  # pragma: no cover
                    raise RuntimeError("ONNX Runtime без CPUExecutionProvider не поддерживается.")
                chosen.append("CPUExecutionProvider")
            session = ort.InferenceSession(str(self.model_path), options, providers=chosen)
            # Fact, not intent: what actually executes the graph.
            self.active_providers = list(session.get_providers())
            shape = session.get_inputs()[0].shape
            if len(shape) != 4 or shape[1] != 3 or shape[2:] != [640, 640]:
                raise RuntimeError("Ожидается YOLOv8 ONNX с входом 1×3×640×640.")
            names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get("names", "{}"))
            if isinstance(names, list):
                names = dict(enumerate(names))
            self.classes = {
                int(k): ALIASES.get(
                    v.lower().replace("_", "-"), ALIASES.get(v.lower(), v.lower())
                )
                for k, v in names.items()
            }
            if not self.required_classes <= set(self.classes.values()):
                missing = sorted(self.required_classes - set(self.classes.values()))
                raise RuntimeError(
                    f"Модель {self.model_path.name} не содержит классы {missing}. "
                    "Ожидаются person, helmet/no_helmet, vest/no_vest."
                )
            self._session = session

    async def predict(self, frame: Frame) -> list[Detection]:
        return await asyncio.to_thread(self.predict_sync, frame)

    def predict_sync(self, frame: Frame) -> list[Detection]:
        # The separate person detector is already needed for worker tracking.
        # In auto mode, zoom PPE inference into each of <=2 visible workers.
        people = self.person_detector.predict_sync(frame) if self.person_detector else []
        if self.roi_mode == "auto" and self.person_detector:
            strong = [d for d in people if d.class_name == "person" and not d.weak]
            height, width = frame.shape[:2]
            if 1 <= len(strong) <= 2:
                rois = [person_roi_pixels(d.bbox, width, height) for d in strong]
                if all(
                    x2 - x1 >= 72 and y2 - y1 >= 72
                    and (x2 - x1) * (y2 - y1) <= 0.8 * width * height
                    for x1, y1, x2, y2 in rois
                ):
                    ppe = []
                    for roi in rois:
                        x1, y1, x2, y2 = roi
                        for det in self._predict_single(frame[y1:y2, x1:x2]):
                            if det.class_name != "person":
                                ppe.append(project_roi_detection(det, roi, width, height))
                    return deduplicate_ppe(ppe) + people
        # Fallback is byte-for-byte the original full-frame inference pathway.
        ppe = self._predict_single(frame)
        if self.person_detector:
            return [d for d in ppe if d.class_name != "person"] + people
        return ppe

    def _predict_single(self, frame: Frame) -> list[Detection]:
        """One forward pass with unchanged 640x640 centered letterbox + decode."""
        import cv2

        self.load()
        height, width = frame.shape[:2]
        scale = min(640 / width, 640 / height)
        nw, nh = round(width * scale), round(height * scale)
        left, top = (640 - nw) // 2, (640 - nh) // 2
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(frame, (nw, nh))
        tensor = (
            np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
            / 255.0
        )
        with self._lock:
            output = self._session.run(None, {self._session.get_inputs()[0].name: tensor})[0]
        return self.decode(output, width, height, scale, left, top)

    def decode(self, output, width, height, scale, left, top) -> list[Detection]:
        import cv2

        rows = output[0].T
        if rows.shape[1] != 4 + len(self.classes):
            raise RuntimeError("Неподдерживаемый формат выхода ONNX (ожидается YOLOv8 без NMS).")
        ids = rows[:, 4:].argmax(axis=1)
        scores = rows[:, 4:].max(axis=1)
        # Display threshold is separated from the violation threshold (F2):
        # keep everything above the floor, flag sub-threshold as weak.
        limits = np.array(
            [self.threshold_for(self.classes.get(int(i), "")) for i in ids],
            dtype=np.float32,
        )
        keep_min = np.minimum(limits, np.full_like(limits, self.floor))
        selected = scores >= keep_min
        rows, scores = rows[selected], scores[selected]
        ids = ids[selected]
        limits = limits[selected]
        keep_min = keep_min[selected]
        detections = []
        for class_id in np.unique(ids):
            class_name = self.classes[int(class_id)]
            if class_name not in KNOWN_CLASSES:
                continue
            mask = ids == class_id
            boxes = rows[mask, :4].copy()
            boxes[:, :2] -= boxes[:, 2:] / 2
            confidences = scores[mask]
            weak_flags = confidences < self.threshold_for(class_name)
            keep = cv2.dnn.NMSBoxes(
                boxes.tolist(), confidences.tolist(), float(keep_min[mask].min()), 0.45
            )
            for index in np.asarray(keep).flatten():
                x, y, w, h = boxes[index]
                coords = np.clip(
                    [
                        (x - left) / scale / width,
                        (y - top) / scale / height,
                        (x + w - left) / scale / width,
                        (y + h - top) / scale / height,
                    ],
                    0,
                    1,
                )
                if coords[2] <= coords[0] or coords[3] <= coords[1]:
                    continue
                detections.append(
                    Detection(
                        class_name,
                        float(confidences[index]),
                        BoundingBox(*map(float, coords)),
                        weak=bool(weak_flags[index]),
                    )
                )
        return detections
