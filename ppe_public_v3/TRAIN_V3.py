"""Train Industrial Safety AI V3 on validated real YOLO boxes. Never deploy automatically.
Run: python ppe_public_v3/TRAIN_V3.py [--epochs 40] [--device auto]
"""
import argparse
import json
import os
from pathlib import Path
import yaml

HERE=Path(__file__).resolve().parent
CLASSES=["person","helmet","no_helmet","vest","no_vest","mask","no_mask"]
# Bare Ultralytics model names honoured for auto-download (clean-machine runs).
KNOWN_PRETRAINED={f'yolo11{n}.pt' for n in 'nsmlx'}|{f'yolov8{n}.pt' for n in 'nsmlx'}

def _check_audit_content(path:Path,strict:bool):
    """An existing filename is not proof: '{}' or garbage must not pass.

    strict (new-builder audits): require the safety verdicts and all-zero
    conflicts/duplicates/drafts. near_similar_pair_count stays a warning
    (similar but distinct scenes are legitimate). Legacy audits predate this
    contract and were validated by the V3 review instead — grandfathered.
    """
    try:
        data=json.loads(path.read_text())
    except (ValueError,OSError) as exc:
        raise SystemExit(f'Dataset audit unreadable: {path} ({exc}). Rebuild the dataset.') from exc
    if not isinstance(data,dict) or 'counts' not in data:
        raise SystemExit(f'Dataset audit invalid: {path} '
                         f'(need a JSON object with "counts"). Rebuild the dataset.')
    if strict:
        for key in ('label_conflict_count','exact_duplicates_across_splits','draft_labels'):
            if key not in data:
                raise SystemExit(f'Dataset audit incomplete: {path} lacks "{key}". '
                                 f'Rebuild with training/build_ppe_v4.py.')
            if data[key]:
                raise SystemExit(f'Dataset audit unsafe: {path} has {key}={data[key]}. '
                                 f'Resolve to zero and rebuild.')
    return path

def resolve_gate(ds:Path):
    """Audit gate file: builders write audit_leaks.json AND audit.json
    (byte-identical); the legacy V3 dir carries AUDIT_VERIFIED.json."""
    if ds == HERE:
        legacy = HERE / 'AUDIT_VERIFIED.json'
        if legacy.is_file():
            print('Dataset audit: legacy V3 validation (grandfathered).',flush=True)
            return _check_audit_content(legacy,strict=False)
    for name in ('audit_leaks.json', 'audit.json'):
        gate = ds / name
        if gate.is_file():
            return _check_audit_content(gate,strict=True)
    raise SystemExit(f'Dataset audit not found in {ds}: '
                     f'need audit_leaks.json (or audit.json).')

def resolve_weights(explicit,project:Path,repo_root:Path):
    """Deterministic base-weights resolution without importing torch.

    Returns (weights, note). A bare known Ultralytics name (e.g. yolo11m.pt,
    with or without ./ prefix) resolves to itself for auto-download on first
    use — so the documented V4 command works on a clean machine given network
    once. Anything path-shaped that is missing fails loudly.
    """
    if explicit:
        w=Path(explicit).expanduser()
        if w.is_file():
            return w.resolve(),'explicit --weights'
        if w.parent==Path('.') and w.name in KNOWN_PRETRAINED:
            return w.name,('Ultralytics auto-download on first use (needs network once; '
                           'or place the file at repo root, or pass an existing --weights path)')
        raise SystemExit(f'Base weights not found: {w}\nPass --weights /full/path/to/base.pt')
    v2best=project/'industrial-ppe-v2'/'weights'/'best.pt'
    if v2best.is_file():
        return v2best,'V2 best.pt'
    local=repo_root/'yolo11m.pt'
    if local.is_file():
        return local,'repo-root yolo11m.pt'
    return 'yolo11m.pt',('Ultralytics auto-download on first use (needs network once; '
                         'or place yolo11m.pt at repo root, or pass --weights explicitly)')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--batch',type=int,default=0,help='0 = choose by GPU')
    p.add_argument('--device',default='auto')
    p.add_argument('--weights',default=None,help='Base checkpoint: existing file, '
                     'or bare Ultralytics name (yolo11m.pt) for auto-download, '
                     'else V2 best.pt when present, else yolo11m.pt')
    p.add_argument('--dataset',default=None,help='Dataset dir with data.yaml (default: this folder)')
    p.add_argument('--resume',action='store_true',help='Resume interrupted V3 run from its own last.pt')
    p.add_argument('--patience',type=int,default=12,help='Early-stopping patience (epochs)')
    p.add_argument('--classes',nargs='+',default=None,
                   help='Override target classes (default: dataset names). '
                        'A non-7-class set needs --weights pointing at a non-PPE base '
                        '(e.g. yolo11m.pt): the detection head resets and decoder/'
                        'thresholds/tests must be updated separately.')
    p.add_argument('--name',default='industrial-ppe-v3')
    args=p.parse_args()
    if args.epochs<1: raise SystemExit('Epochs must be >=1')
    ds = Path(args.dataset).expanduser().resolve() if args.dataset else HERE
    gate = resolve_gate(ds)
    print(f'Dataset audit: {gate}',flush=True)
    info=yaml.safe_load((ds/'data.yaml').read_text())
    target = list(args.classes) if args.classes else list(info['names'])
    if list(info['names'])!=target:
        raise SystemExit(f'Dataset names {list(info["names"])} != --classes {target}; '
                         'rebuild the dataset with the builder --target-classes first.')
    info['path']=str(ds)
    (ds/'data.yaml').write_text(yaml.safe_dump(info,sort_keys=False))
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit('Install: python -m pip install ultralytics PyYAML') from e
    if args.device=='auto':
        device='0' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu')
    else: device=args.device
    batch=args.batch or (8 if device=='0' else (4 if device=='mps' else 2))
    project=HERE.parent.resolve()/'training'/'runs'
    if args.resume:
        weights=project/'industrial-ppe-v3'/'weights'/'last.pt'
        if not weights.is_file(): raise SystemExit(f'Nothing to resume: {weights}')
        resume, exist_ok, run_name = True, True, 'industrial-ppe-v3'
    else:
        resume, exist_ok, run_name = False, False, args.name
    if not args.resume:
        weights,note=resolve_weights(args.weights,project,HERE.parent.resolve())
        print(f'Base weights: {weights} ({note}).',flush=True)
    try:
        model=YOLO(str(weights))
    except Exception as exc:
        raise SystemExit(f'Cannot load base weights {weights}: {exc}\n'
                         f'Place yolo11m.pt at {HERE.parent.resolve()} or pass --weights.') from exc
    if list(model.names.values())!=target:
        if len(model.names)==len(target):
            raise SystemExit('PPE class order differs from dataset; abort. '
                             'For a new class set, pass a non-PPE base via --weights '
                             'and update decoder/thresholds/tests separately.')
        print(f'Transfer learning: {len(model.names)}-class base, detection head resets to {len(target)} classes.',flush=True)
    print('Dataset:',ds,'GPU:',device,'Epochs:',args.epochs,'Batch:',batch,flush=True)
    model.train(data=str(ds/'data.yaml'),epochs=args.epochs,imgsz=640,batch=batch,
                device=device,workers=0 if device=='mps' else 4,
                project=str(project),name=run_name,exist_ok=exist_ok,resume=resume,
                patience=args.patience,seed=17,cache=False,verbose=True)
    best=Path(model.trainer.save_dir)/'weights'/'best.pt'
    if not best.is_file(): raise SystemExit('Training finished with no best.pt')
    print('Best model:',best)
    print('Do not change models/ppe.onnx until independent CCTV validation.')

if __name__=='__main__':main()
