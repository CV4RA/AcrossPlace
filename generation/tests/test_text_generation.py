from text_generation.visual_parser import VisualFacts
from text_generation.place_fusion import fuse_visual_facts
from text_generation.quality_filter import filter_caption
import pytest

def test_schema():
    with pytest.raises(ValueError): VisualFacts.model_validate({'scene_type':'road'})

def test_fusion():
    v={'scene_type':'road','static_objects':[{'class':'building','attributes':['red'],'relative_position':'left'}],'spatial_relations':[]}
    f=fuse_visual_facts({'a':v,'b':v})['facts']
    assert len([x for x in f if x['kind']=='object'])==1
    assert len([x for x in f if x['kind']=='view_object'])==2
    assert all(x['support']==1 for x in f if x['kind']=='object')

def test_filter():
    class Verifier:
        def generate(self,prompt): return {'grounded':True,'consistent':True,'unsupported_claims':[]}
    f={'facts':[{'fact_id':'F0','value':{'class':'building'},'kind':'object','source_images':['a']}]}
    p={'place_id':'x:1','metadata':{'city':'London'}}
    t={'text':'A red building stands in London.','source_fact_ids':['F0'],'claims':['red building']}
    r=filter_caption(t,f,p,Verifier(),{},set(),{},10)
    assert not r['accepted'] and 'metadata_leak' in r['filter_reasons']


def test_four_categories_synthetic_generator():
    import re
    from text_generation.caption_generator import generate_captions
    class FakeVLM:
        def generate(self,prompt):
            if prompt.startswith('Generate'):
                count,kind=re.search(r'Generate (\d+) distinct (\w+)',prompt).groups()
                return {'captions':[{'text':f'A red brick building beside a road, {kind} fixture {i}.','source_fact_ids':['F0'],'claims':['red brick building']} for i in range(int(count))]}
            return {'grounded':True,'consistent':True,'unsupported_claims':[]}
    fused={'facts':[{'fact_id':'F0','kind':'object','value':{'class':'building','attributes':['red','brick']},'source_images':['/City_10.12345_20.12345.jpg']}]}
    captions=generate_captions(FakeVLM(),fused,{'per_type':2})
    assert len(captions)==8
    assert {c['type'] for c in captions}=={'detailed','concise','partial','relational'}
    from text_generation.prompts import prompt_facts
    assert 'City_' not in str(prompt_facts(fused['facts']))

def test_parser_retry():
    from text_generation.visual_parser import parse_visual
    class FakeVLM:
        def __init__(self): self.calls=0
        def generate(self,*args):
            self.calls+=1
            if self.calls==1: return {'scene_type':12}
            return {'scene_type':'road','static_objects':[{'category':'road','attributes':[],'relative_position':'center'}],'spatial_relations':[],'transient_objects':[],'environment':[],'uncertain_facts':[]}
    v=FakeVLM(); assert parse_visual(v,'synthetic-path')['scene_type']=='road'; assert v.calls==2
