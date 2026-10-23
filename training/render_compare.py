"""Render orig/v1/v2 predictions on the SAME frames for visual audit.

Usage:
    .venv/bin/python training/render_compare.py

Writes audit_compare/compare_<frame>__orig|v1|v2.jpg (annotated singles),
audit_compare/triptych_<frame>.jpg (side-by-side) and audit_compare/SUMMARY.md.
Uses production provider settings (per-class thresholds, NMS, ROI off) so the
pictures match what the app would draw, minus tracker overlays.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))

import cv2

from app.core.config import settings
from app.inference.local_onnx import LocalOnnxProvider

MODELS = {
    'orig': ROOT / 'models' / 'ppe.onnx',
    'v1': ROOT / 'training' / 'runs' / 'industrial-ppe-v1' / 'weights' / 'best.onnx',
    'v2': ROOT / 'models' / 'ppe-candidate.onnx',
}

SNAPSHOTS = [
    'data/snapshots/0cec764a84cf48758cdf3103e345a7c3.jpg',
    'data/snapshots/49bfd12c5e694e0a87f60d9f8946d134.jpg',
    'data/snapshots/53c715e8b24c42c3bd46939ac81a5a25.jpg',
    'data/snapshots/66ee6da37ea3431db3f137b25fc5c8c8.jpg',
]
TEST_FRAMES = ['coco_000016', 'coco_000017', 'coco_000013', 'rf100_000034']

COLORS = {
    'person': (255, 200, 60), 'helmet': (60, 220, 60), 'no_helmet': (60, 60, 240),
    'vest': (60, 220, 60), 'no_vest': (60, 60, 240), 'mask': (60, 220, 60),
    'no_mask': (60, 60, 240), 'smoking': (60, 60, 240),
}


def annotate(frame, detections, tag):
    out = frame.copy()
    h, w = out.shape[:2]
    cv2.putText(out, tag, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    for d in detections:
        color = COLORS.get(d.class_name, (200, 200, 200))
        x1, y1, x2, y2 = (int(d.bbox.x1 * w), int(d.bbox.y1 * h),
                          int(d.bbox.x2 * w), int(d.bbox.y2 * h))
        thick = 1 if d.weak else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(out, f"{d.class_name} {d.confidence:.2f}{' weak' if d.weak else ''}",
                    (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main() -> None:
    out = ROOT / 'audit_compare'
    out.mkdir(exist_ok=True)
    providers = {tag: LocalOnnxProvider(str(path), person_model_path=str(ROOT / 'models' / 'person.onnx'),
                                        thresholds=settings.class_thresholds, floor=settings.display_floor,
                                        roi_mode='off')
                 for tag, path in MODELS.items()}
    frames: list[tuple[str, Path]] = (
        [(f'snap{i}', ROOT / s) for i, s in enumerate(SNAPSHOTS)]
        + [(f'test_{t}', ROOT / 'datasets' / 'ppe_public_v1' / 'test' / 'images' / f'{t}.jpg')
           for t in TEST_FRAMES]
    )
    summary = ['# orig/v1/v2 visual comparison', '',
               'Solid box = above class threshold. Thin box + weak = seen but below threshold.',
               'Same frame, same provider settings (thresholds/NMS as production, ROI off).', '']
    for name, path in frames:
        frame = cv2.imread(str(path))
        if frame is None:
            print('SKIP unreadable:', path)
            continue
        rendered = {}
        det_lines = []
        for tag, provider in providers.items():
            dets = provider.predict_sync(frame)
            rendered[tag] = annotate(frame, dets, tag)
            cv2.imwrite(str(out / f'compare_{name}__{tag}.jpg'), rendered[tag],
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            shown = ', '.join(f'{d.class_name} {d.confidence:.2f}{"w" if d.weak else ""}' for d in dets) or 'nothing'
            det_lines.append(f'- {tag}: {shown}')
        triptych = cv2.hconcat([rendered['orig'], rendered['v1'], rendered['v2']])
        cv2.imwrite(str(out / f'triptych_{name}.jpg'), triptych, [cv2.IMWRITE_JPEG_QUALITY, 85])
        summary += [f'## {name} ({path.name})', *det_lines, '']
        print('done', name)
    (out / 'SUMMARY.md').write_text('\n'.join(summary), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
