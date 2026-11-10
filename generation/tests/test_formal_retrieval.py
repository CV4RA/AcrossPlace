"""Formal-data provenance and identity guards."""
import copy,json
from pathlib import Path
import pytest
from tests.helpers import synthetic_annotations
from utils.io import read_jsonl,write_jsonl
from data.formal_retrieval import prepare_formal

def formal_fixture(tmp_path):
    source=Path(synthetic_annotations(tmp_path/'images')); rows=read_jsonl(source)
    for i,row in enumerate(rows):
        row['split']=['train','train','val','val','test','test'][i]
        for j,t in enumerate(row['texts']): t['quality_status']='valid' if j%2==0 else 'uncertain'
    write_jsonl(source,rows)
    c={'annotations':str(source),'output':str(tmp_path/'formal'),'verify_near_duplicates':False}
    return c,rows

def test_formal_data_freezes_protocol_and_quality_ablation(tmp_path):
    c,original=formal_fixture(tmp_path); m=prepare_formal(c); strict=read_jsonl(Path(c['output'])/'valid.jsonl'); expanded=read_jsonl(Path(c['output'])/'valid_plus_uncertain_same_places.jsonl')
    assert m['splits']['train']['places']==2 and m['splits']['train']['texts']==4
    assert all(t['quality_status']=='valid' for r in strict for t in r['texts'])
    assert [r['place_id'] for r in strict]==[r['place_id'] for r in expanded]
    assert all(a['source_image_paths']==b['source_image_paths'] and a['target_image_paths']==b['target_image_paths'] for a,b in zip(original,strict))
    assert [r for r in strict if r['split']!='train']==[r for r in expanded if r['split']!='train']
    assert prepare_formal({**c,'prepared_only':True,'expected_fingerprint':m['fingerprint']})==m
    (Path(c['output'])/'test_t2i_gallery.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='modified'): prepare_formal(c)

def test_formal_data_rejects_source_gallery_leakage(tmp_path):
    c,rows=formal_fixture(tmp_path); rows[0]['target_image_paths']=rows[0]['source_image_paths']; write_jsonl(c['annotations'],rows)
    with pytest.raises(ValueError,match='Source/gallery overlap'): prepare_formal(c)
