import pytest
from tests.helpers import synthetic_annotations
from utils.io import read_jsonl
from eval.protocols import select_protocol

def test_protocols(tmp_path):
    rows=read_jsonl(synthetic_annotations(tmp_path))
    selected,skipped=select_protocol(rows,'B',0); assert len(selected)==6
    assert all(not set(r['evaluation_image_paths'])&set(r['source_image_paths']) for r in selected)
    selected,_=select_protocol(rows,'C',0); assert len(selected)==6
    rows[0]['target_image_paths']=rows[0]['source_image_paths']
    with pytest.raises(ValueError): select_protocol(rows,'B',0)

def test_missing_conditions(tmp_path):
    rows=read_jsonl(synthetic_annotations(tmp_path)); del rows[0]['images'][0]['year']
    with pytest.raises(ValueError): select_protocol(rows,'C',0)
