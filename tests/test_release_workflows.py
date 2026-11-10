"""Guard portability, frozen choices and a real local-query encoding round trip."""
import importlib.util
import json
from pathlib import Path
import pytest
from place_retrieval.io import digest,write_json,sha256
from place_retrieval.study.plan import make_plan

def module(name):
    p=Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_study_freeze_rejects_changed_checkpoint(tmp_path):
    m=module('study_test');root=tmp_path/'study';out=root/'runs/a';out.mkdir(parents=True)
    config=root/'config.json';write_json(config,{'seed':42});c={'seed':42}
    write_json(out/'job_complete.json',{'config_digest':digest(c)})
    (out/'coarse').mkdir();(out/'coarse/best.pt').write_bytes(b'original')
    write_json(out/'coarse/val_result.json',{'split':'val'})
    write_json(root/'plan.json',{'fingerprint':'plan','jobs':[{'config':str(config),'config_digest':digest(c),'output':str(out),'action':'zero_coarse'}]})
    m.freeze(root);m.verify_gate(root)
    (out/'coarse/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen artifact changed'):m.verify_gate(root)

def test_local_index_rejects_different_checkpoint(tiny_config,tmp_path):
    from place_retrieval.training import train
    from types import SimpleNamespace
    m=module('retrieve');c=tiny_config;checkpoint=train(c)
    config=tmp_path/'config.json';write_json(config,c);index=tmp_path/'index.pt'
    m.run(SimpleNamespace(command='build-index',config=str(config),checkpoint=str(checkpoint),index=str(index),split='test'))
    m.run(SimpleNamespace(command='query',config=str(config),checkpoint=str(checkpoint),index=str(index),text='red building',image=None,top_k=2))
    with checkpoint.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='identity differs'):
        m.run(SimpleNamespace(command='query',config=str(config),checkpoint=str(checkpoint),index=str(index),text='red building',image=None,top_k=2))
