import json,threading
from pathlib import Path
from tests.test_place_pipeline import settings,FakeVLM
from text_generation.vllm_pipeline import ConcurrentPlacePipeline
from text_generation.batch_tasks import caption_task
from utils.io import read_jsonl

class FakeClient:
    def __init__(self): self.calls=0; self.lock=threading.Lock(); self.failed_once=False
    def complete(self,task):
        with self.lock: self.calls+=1
        if task.get('image'):
            value=FakeVLM().generate('visual',task['image'])
        else:
            kind=task['kind']; facts=json.loads(task['prompt'].split('Evidence: ',1)[1]); view=next((f['value']['view'] for f in facts if f['kind']=='relation'),None)
            descriptions={'detailed':['A red brick building is visible beside a narrow road.','From this viewpoint, the red brick building stands beside the road.'],'concise':['A red brick building stands here.','The narrow road is visible.'],'partial':['A red brick building is visible.','A narrow road runs here.'],'relational':['From this viewpoint, the red brick building is beside the narrow road.','From this camera view, a building stands beside the road.']}
            value={'captions':[{'text':t,'source_fact_ids':task['ids'],'view_id':view if kind=='relational' else None} for t in descriptions[kind]]}
        return {'value':value,'raw_text':json.dumps(value),'usage':{'completion_tokens':10,'prompt_tokens':30},'elapsed_s':.01,'finish_reason':'stop'}

def batch_config(tmp_path):
    c=settings(tmp_path); c['model_identity']={'name':'SYNTHETIC-TEST-DOUBLE','revision':'fixture'}; c['vllm']={'concurrency':3,'served_model':'fixture','base_url':'http://127.0.0.1:1/v1'}
    c['protocol'].update(seed=42,max_source_views=1); return c

def test_cross_place_batch_resume_and_quality(tmp_path):
    c=batch_config(tmp_path); client=FakeClient(); p=ConcurrentPlacePipeline(c,client); plans=p.prepare(); fused=p.extract(plans); raw=p.generate(fused); q=p.filter(raw); p.close()
    assert len(raw)==6 and q['generated_texts']==48 and q['retained_texts']>0
    assert all(len(r['texts'])==8 for r in raw)
    before=client.calls; p=ConcurrentPlacePipeline(c,client); fused=p.extract(plans); assert len(p.generate(fused))==6; p.close(); assert client.calls==before
    sources={im['path'] for plan in plans for im in plan['source_images']}; targets={im['path'] for plan in plans for im in plan['gallery_images']}; assert not sources&targets

def test_retry_failed_job_is_not_replaced_with_fake_labels(tmp_path):
    c=batch_config(tmp_path); client=FakeClient(); original=client.complete; attempts={}
    def flaky(task):
        attempts[task['key']]=attempts.get(task['key'],0)+1
        if attempts[task['key']]==1: raise RuntimeError('temporary backend error')
        return original(task)
    client.complete=flaky; p=ConcurrentPlacePipeline(c,client); plans=p.prepare(limit=1); rows=p.extract(plans); assert rows and not p.failures; assert all(n==2 for n in attempts.values()); p.close()

def test_caption_schema_enforces_count_and_legal_fact_ids(tmp_path):
    c=batch_config(tmp_path); p=ConcurrentPlacePipeline(c,FakeClient()); fused=p.extract(p.prepare(limit=1)); prompt,schema,ids,views=caption_task(fused[0]['fusion'],'relational',c['generation']); p.close()
    assert schema['properties']['captions']['minItems']==2 and schema['properties']['captions']['maxItems']==2
    assert schema['$defs']['Caption']['properties']['source_fact_ids']['items']['enum']==ids
    assert 'claims' not in schema['$defs']['Caption']['properties']


def test_frozen_plan_preserves_membership_and_rejects_overlap(tmp_path):
    import pytest
    from utils.io import write_jsonl
    c=batch_config(tmp_path); p=ConcurrentPlacePipeline(c,FakeClient()); plans=p.prepare(); p.close()
    frozen=tmp_path/'frozen.jsonl'; write_jsonl(frozen,plans)
    c['dataset']['plan_file']=str(frozen); c['output']['directory']=str(tmp_path/'pilot')
    p=ConcurrentPlacePipeline(c,FakeClient()); pilot=p.prepare(limit=1); p.close()
    assert len(pilot)==1
    assert pilot[0]['source_images']==plans[0]['source_images']
    assert pilot[0]['gallery_images']==plans[0]['gallery_images']
    plans[0]['gallery_images'].append(plans[0]['source_images'][0]); write_jsonl(frozen,plans)
    p=ConcurrentPlacePipeline(c,FakeClient())
    try:
        with pytest.raises(ValueError,match='source/gallery overlap'): p.prepare(limit=1)
    finally: p.close()
