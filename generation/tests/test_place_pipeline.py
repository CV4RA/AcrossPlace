"""All generated content here is synthetic test-double output, never real labels."""
import json,copy
from pathlib import Path
from tests.helpers import synthetic_annotations
from utils.io import read_jsonl,write_jsonl,config
from text_generation.pipeline import PlaceTextPipeline
from data.crossmodal_builder import build_crossmodal_dataset

class FakeVLM:
    images=[]; loads=0; generations=0
    def __init__(self,*args):
        type(self).loads+=1; self.info={'model':'SYNTHETIC-TEST-DOUBLE','revision':'fixture'}; self.last_responses=[]
    def generate(self,prompt,image=None):
        type(self).generations+=1
        if image:
            type(self).images.append(image)
            return {'scene_type':'street','static_objects':[{'object_id':'b','category':'building','attributes':['red','brick'],'relative_position':'left','confidence':.9},{'object_id':'r','category':'road','attributes':['narrow'],'relative_position':'center','confidence':.9}],'spatial_relations':[{'subject':'b','predicate':'beside','object':'r'}],'transient_objects':['car'],'environment':{'illumination':'daytime','weather':'clear'},'uncertain_facts':[]}
        import re
        count,kind=re.search(r'Generate (\d+) distinct (\w+)',prompt).groups(); facts=json.loads(prompt.split('Evidence: ',1)[1]); view=next((f['value']['view'] for f in facts if f['kind']=='relation'),None)
        relevant=[f for f in facts if f['value'].get('view')==view] or facts
        sentences={'detailed':['A red brick building and a narrow road are visible.','The narrow road has a red brick building beside it.'],'concise':['A brick building stands beside the road.','A narrow road near a red building.'],'partial':['The red brick building is visible.','The narrow road is visible here.'],'relational':['From this viewpoint, a red brick building is beside a narrow road.','From the camera view, the narrow road lies beside the brick building.']}
        return {'captions':[{'text':sentences[kind][i%2],'source_fact_ids':[f['fact_id'] for f in relevant],'claims':[sentences[kind][i%2]],'view_id':view if kind=='relational' else None} for i in range(int(count))]}

def settings(tmp_path):
    annotations=synthetic_annotations(tmp_path/'fixtures'); rows=read_jsonl(annotations)
    for r in rows:
        for im in r['images']:
            from utils.io import digest
            im['image_id']='I_'+digest(im['path'])[:24]
    write_jsonl(annotations,rows); c=config(Path(__file__).resolve().parents[1]/'configs/text_generation.yaml'); c['dataset']['index']=annotations; c['dataset']['max_places']=6; c['dataset']['max_views_per_place']=1
    c['protocol']['duplicate_hamming']=0; c['output']={'directory':str(tmp_path/'out')}; return c

def test_end_to_end_sources_resume_reports(tmp_path):
    FakeVLM.images=[]; FakeVLM.loads=0; FakeVLM.generations=0; c=settings(tmp_path)
    p=PlaceTextPipeline(c,vlm_factory=FakeVLM)
    fused=p.extract(); rows=p.generate(); report=p.filter(); p.close()
    assert len(rows)==6 and all(len(r['texts'])==8 for r in rows)
    assert FakeVLM.loads==1
    assert not set(FakeVLM.images)&{path for r in fused for path in r['target_image_paths']}
    assert report['generated_texts']==48 and report['retained_texts']>0
    before=FakeVLM.generations; second=PlaceTextPipeline(c,vlm_factory=FakeVLM); second.extract(); restored=second.generate(); assert len(restored)==6; second.close()
    assert FakeVLM.generations==before and FakeVLM.loads==1
    built=build_crossmodal_dataset({'annotations':str(tmp_path/'out/annotations/place_texts_filtered.jsonl'),'output':str(tmp_path/'dataset'),'duplicate_hamming':0})
    assert built['exported_places']==6 and built['query_images']==6
    metrics=json.loads((tmp_path/'out/reports/generation_metrics.json').read_text()); assert not metrics['extract']['model_inference_executed']; assert metrics['extract']['peak_gpu_memory_bytes'] is None

def test_dry_run_does_not_load_model(tmp_path):
    c=settings(tmp_path); before=FakeVLM.loads; p=PlaceTextPipeline(c,vlm_factory=FakeVLM); p.extract(dry_run=True); p.close(); assert FakeVLM.loads==before

def test_failed_visual_is_logged_no_fake_label(tmp_path):
    class Bad(FakeVLM):
        def generate(self,*args): return {'bad':'invalid'}
    c=settings(tmp_path); c['generation']['max_retries']=0; p=PlaceTextPipeline(c,limit_places=1,vlm_factory=Bad)
    assert p.extract()==[]; assert p.failures
    records=read_jsonl(tmp_path/'out/annotations/image_facts.jsonl'); assert records and all(r['status']=='failed' and r['facts'] is None for r in records); p.close()


def test_cached_annotations_follow_current_split(tmp_path):
    c=settings(tmp_path); p=PlaceTextPipeline(c,vlm_factory=FakeVLM); p.extract(); p.generate(); p.close()
    prepared=read_jsonl(c['dataset']['index']); prepared[0]['split']='val'; write_jsonl(c['dataset']['index'],prepared)
    before=FakeVLM.generations; p=PlaceTextPipeline(c,vlm_factory=FakeVLM); p.extract(); rows=p.generate(); p.close()
    assert len(rows)==6 and rows[0]['split']=='val'
    images=read_jsonl(tmp_path/'out/annotations/image_facts.jsonl')
    assert all(r['split']=='val' for r in images if r['place_id']==prepared[0]['place_id'])
    assert FakeVLM.generations==before
