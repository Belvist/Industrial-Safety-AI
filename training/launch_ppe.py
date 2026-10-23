"""One command: download labeled PPE data, validate, fine-tune, test, export ONNX.

After installing Python 3.11 and the pinned dependencies:
    python training/launch_ppe.py --device 0       # NVIDIA GPU
    python training/launch_ppe.py --device mps     # Apple Silicon
    python training/launch_ppe.py --dry-run        # print required steps only

Dataset/source weights subject to CC BY 4.0 and Ultralytics AGPL-3.0 respectively.
Does NOT deploy candidate weights without a real held-out camera evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))

from build_public_ppe import CLASSES, assemble, download_sources, load_coco, load_rf  # noqa: E402


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(args: argparse.Namespace) -> dict:
    output = (ROOT / args.output).resolve()
    if not (output / 'data.yaml').exists():
        rf, coco = download_sources(ROOT / 'datasets' / '.source_cache')
        stats = assemble(load_rf(rf) + load_coco(coco), output)
    else:
        stats = json.loads((output / 'audit.json').read_text())
    # A dataset packed/uploaded on another machine may carry an old absolute
    # path. Refresh it at the new machine location before training.
    import yaml
    config_path = output / 'data.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['path'] = str(output)
    if config.get('names') != CLASSES:
        raise RuntimeError(f'Incompatible dataset class mapping: {config.get("names")}')
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print('Dataset ready:', output)
    print('Images:', stats['counts'], 'Class boxes:', stats['per_class'])
    if args.archive:
        archive = output.with_suffix('.zip')
        print('Packaging labeled images, YOLO txt, audit and credits:', archive)
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as bundle:
            for path in sorted(output.rglob('*')):
                if path.is_file():
                    bundle.write(path, arcname=str(path.relative_to(output.parent)))
        print('Archive SHA256:', file_sha256(archive))
    if args.prepare_only:
        return {'dataset': str(output), 'status': 'prepared', 'audit': stats}
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install training packages: pip install -r training/requirements-train.txt') from exc
    if args.device == '0' and not torch.cuda.is_available():
        raise SystemExit('CUDA device 0 unavailable; choose --device mps or --device cpu.')
    if args.device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise SystemExit('MPS unavailable; choose --device 0 or --device cpu.')

    print('Starting fine-tuning. No claims of accuracy until validation on real footage.')
    if args.resume:
        ckpt = ROOT / 'training' / 'runs' / 'industrial-ppe-v1' / 'weights' / 'last.pt'
        if not ckpt.is_file():
            raise SystemExit(f'Nothing to resume: {ckpt} missing')
        model = YOLO(str(ckpt))
        run_name, resume, exist_ok = 'industrial-ppe-v1', True, True
    elif args.warm_start:
        # Post-completion continuation: ultralytics strips optimizer state on
        # clean finish, so resume=True is impossible; warm-start instead.
        ckpt = Path(args.warm_start)
        if not ckpt.is_file():
            raise SystemExit(f'Warm-start weights missing: {ckpt}')
        model = YOLO(str(ckpt))
        run_name, resume, exist_ok = 'industrial-ppe-v2', False, False
    else:
        model = YOLO(args.model)
        run_name, resume, exist_ok = 'industrial-ppe-v1', False, False
    started = time.time()
    results = model.train(
        data=str(output / 'data.yaml'),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / 'training' / 'runs'),
        name=run_name,
        exist_ok=exist_ok,
        resume=resume,
        patience=12,
        seed=17,
        cache=False,
        verbose=True,
    )
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / 'weights' / 'best.pt'
    if not best.is_file():
        raise RuntimeError('Training finished without best.pt')
    validated = YOLO(str(best))
    if list(validated.names.values()) != CLASSES:
        raise RuntimeError(f'Unexpected class map in trained weights: {validated.names}')
    # Report held-out test exactly once, not a tuning criterion.
    test_metrics = validated.val(data=str(output / 'data.yaml'), split='test', imgsz=640,
                                 device=args.device, project=str(save_dir), name='heldout-test')
    recorded = {'classes': CLASSES, 'best_pt': str(best), 'best_sha256': file_sha256(best),
                'training_seconds': round(time.time() - started),
                'dataset': str(output), 'data_distribution': stats['per_class'],
                'test_map50': float(test_metrics.box.map50),
                'test_map50_95': float(test_metrics.box.map),
                'test_precision': float(test_metrics.box.mp),
                'test_recall': float(test_metrics.box.mr),
                'warning': 'Test metrics are NOT industrial video/event-level accuracy. Hold-out local site videos are required.'}
    if not args.skip_export:
        print('Exporting candidate ONNX; leaving deployed models/ppe.onnx unchanged.')
        onnx = Path(validated.export(format='onnx', imgsz=640, opset=12,
                                      dynamic=False, simplify=False, nms=False, device='cpu'))
        target = ROOT / 'models' / 'ppe-candidate.onnx'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx, target)
        # Check model input/output + metadata against production reader contract.
        import ast
        import onnxruntime as ort
        session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
        dims = list(session.get_inputs()[0].shape)
        if dims != [1, 3, 640, 640]:
            target.unlink(missing_ok=True)
            raise RuntimeError(f'Unsupported ONNX input dimensions: {dims}')
        names = ast.literal_eval(session.get_modelmeta().custom_metadata_map.get('names', '{}'))
        exported_names = list(names.values()) if isinstance(names, dict) else list(names)
        if exported_names != CLASSES:
            target.unlink(missing_ok=True)
            raise RuntimeError(f'Incorrect ONNX class map: {exported_names}')
        outputs = session.get_outputs()
        if not outputs or len(outputs[0].shape) != 3:
            target.unlink(missing_ok=True)
            raise RuntimeError('Unsupported ONNX tensor shape')
        recorded['onnx_candidate'] = str(target)
        recorded['onnx_sha256'] = file_sha256(target)
        (target.parent / (target.name + '.sha256')).write_text(recorded['onnx_sha256'] + '  ' + target.name + '\n')
    (save_dir / 'industrial_eval.json').write_text(json.dumps(recorded, indent=2, ensure_ascii=False))
    return recorded


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('datasets/ppe_public_v1'))
    p.add_argument('--model', default='yolo11m.pt', help='Pretrained YOLO checkpoint; check AGPL-3.0')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='0', help='0 CUDA, mps Mac, cpu')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--archive', action='store_true', help='Produce dataset ZIP for Cloudflare backup')
    p.add_argument('--skip-export', action='store_true')
    p.add_argument('--resume', action='store_true',
                   help='Resume interrupted run from training/runs/industrial-ppe-v1/weights/last.pt')
    p.add_argument('--warm-start', default=None,
                   help='Warm-start from a finished checkpoint (e.g. v1 best.pt) into a new industrial-ppe-v2 run')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        p.error('Invalid epochs, batch or workers')
    if args.dry_run:
        print('Dataset sources: LibreYOLO/construction-safety-gsnvb + keremberke/construction-safety-object-detection')
        print('Class order: ' + ', '.join(CLASSES))
        print('Input label types: YOLO and COCO, converted with explicit mapping')
        print('Training plan: ' + str(args.epochs) + ' epochs, ' + args.model + ', 640px, ' + args.device)
        print('Output: weights/best.pt, models/ppe-candidate.onnx (not automatically deployed)')
        return
    run(args)


if __name__ == '__main__':
    main()
