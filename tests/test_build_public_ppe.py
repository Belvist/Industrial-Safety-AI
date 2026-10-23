from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / 'training'))
from build_public_ppe import CLASSES, RF_REMAP, assemble, load_coco, load_rf  # noqa: E402


def small_fixture(tmp_path):
    rf, coco = tmp_path / 'rf', tmp_path / 'coco'
    import yaml
    rf.mkdir()
    (rf / 'data.yaml').write_text(yaml.safe_dump({'names': ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']}))
    cats = ['person', 'hardhat', 'no-hardhat', 'safety vest', 'no-safety vest', 'mask', 'no-mask', 'unrelated']
    for i, split in enumerate(['train', 'valid', 'test']):
        p = rf / split
        for sub in ['images', 'labels']:
            (p / sub).mkdir(parents=True)
        Image.new('RGB', (64,64), (i*50,20,30)).save(p/'images'/'img.jpg')
        (p/'labels'/'img.txt').write_text('1 0.5 0.5 0.2 0.2\n2 0.5 0.65 0.2 0.3\n')
        target = coco / split
        target.mkdir(parents=True)
        Image.new('RGB', (64,64), (20,i*50,60)).save(target/'x.jpg')
        entries = [{'id':idx,'name':name} for idx,name in enumerate(cats)]
        obj = {'categories': entries, 'images':[{'id':1,'file_name':'x.jpg','width':64,'height':64}],
               'annotations': [{'id':1,'image_id':1,'category_id':5,'bbox':[16,12,20,20]}]}
        (target/'_annotations.coco.json').write_text(json.dumps(obj))
    return rf,coco


def test_class_map_and_build(tmp_path):
    rf, coco = small_fixture(tmp_path)
    assert RF_REMAP == {0:1,1:2,2:4,3:0,4:3}
    records = load_rf(rf) + load_coco(coco)
    result = assemble(records, tmp_path/'out')
    assert result['counts'] == {'test':2,'val':2,'train':2}
    assert result['per_class']['no_helmet']==3
    assert result['per_class']['no_vest']==3
    assert result['per_class']['mask']==3
    assert result['labels_total']==9
    data = (tmp_path/'out'/'data.yaml').read_text()
    assert 'no_helmet' in data and 'no_vest' in data and 'no_mask' in data
    for split in ('train','val','test'):
        assert len(list((tmp_path/'out'/split/'images').glob('*.jpg'))) == 2
        assert len(list((tmp_path/'out'/split/'labels').glob('*.txt'))) == 2


def test_dedup_never_crosses_splits(tmp_path):
    rf,coco=small_fixture(tmp_path)
    # Copy train frame to test. Test has priority, so duplicate train is dropped.
    src=rf/'test/images/img.jpg'
    (rf/'train/images/img.jpg').write_bytes(src.read_bytes())
    result=assemble(load_rf(rf)+load_coco(coco),tmp_path/'out')
    assert result['removed_exact_duplicate_images'] == 1
    assert result['counts']['test']==2
    assert result['counts']['train']==1


def test_reject_invalid_bbox(tmp_path):
    rf,coco=small_fixture(tmp_path)
    (rf/'train/labels/img.txt').write_text('1 0.99 0.99 0.5 0.5\n')
    with pytest.raises(ValueError):
        load_rf(rf)


def test_clip_rounding_edge(tmp_path):
    # x+w/2 = 1.0009: pixel->normalized rounding, must clip, not crash.
    rf,coco=small_fixture(tmp_path)
    (rf/'train/labels/img.txt').write_text('1 0.9692 0.4047 0.0634 0.0354\n')
    records=load_rf(rf)
    boxes=[c for _,c in records[0].labels]
    assert boxes and max(b[0]+b[2]/2 for b in boxes) <= 1.0
