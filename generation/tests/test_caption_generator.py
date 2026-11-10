import json,re
from text_generation.caption_generator import generate_captions

def test_partial_subset_and_relational_view_binding():
    facts=[{'fact_id':'F0','kind':'view_object','value':{'class':'building','attributes':['red'],'relative_position':'left','view':'/secret/City/GPS_A.jpg'},'source_images':['/secret/City/GPS_A.jpg']},{'fact_id':'F1','kind':'view_object','value':{'class':'building','attributes':['glass'],'view':'/secret/City/GPS_B.jpg'},'source_images':['/secret/City/GPS_B.jpg']},{'fact_id':'F2','kind':'relation','value':{'subject':'building','predicate':'beside','object':'road','view':'/secret/City/GPS_B.jpg'},'source_images':['/secret/City/GPS_B.jpg']},{'fact_id':'F3','kind':'object','value':{'class':'building','attributes':['red']},'source_images':['/secret/City/GPS_A.jpg']}]
    class Recorder:
        def __init__(self): self.evidence={}
        def generate(self,prompt):
            assert '/secret' not in prompt and 'GPS_' not in prompt
            count,kind=re.search(r'Generate (\d+) distinct (\w+)',prompt).groups(); evidence=json.loads(prompt.split('Evidence: ',1)[1]); self.evidence[kind]=evidence
            view=next((f['value']['view'] for f in evidence if f['kind']=='relation'),None)
            return {'captions':[{'text':'A visible red building stands here '+str(i),'source_fact_ids':[f['fact_id'] for f in evidence],'claims':['visible building'],'view_id':view} for i in range(int(count))]}
    generator=Recorder(); rows=generate_captions(generator,{'facts':facts},{'per_type':2})
    assert all(f['kind']=='object' for f in generator.evidence['partial'])
    assert all(t['bound_view']=='/secret/City/GPS_B.jpg' for t in rows if t['type']=='relational')


def test_incremental_count_and_optional_claims_are_cached(tmp_path):
    from utils.generation_cache import GenerationCache
    facts={'facts':[{'fact_id':'F0','kind':'object','value':{'class':'building','attributes':['red']},'source_images':['a']}]}
    class Single:
        def __init__(self): self.calls=0
        def generate(self,prompt):
            self.calls+=1
            return {'captions':[{'text':f'A red building, variant {self.calls}.','source_fact_ids':['F0'],'claims':[]}]}
    model=Single(); cache=GenerationCache(tmp_path/'cache.sqlite')
    rows=generate_captions(model,facts,{'types':['concise'],'per_type':2},cache,'fixture')
    assert len(rows)==2 and model.calls==2
    assert generate_captions(model,facts,{'types':['concise'],'per_type':2},cache,'fixture')==rows
    assert model.calls==2; cache.close()


def test_partial_retry_moves_to_unused_grounded_object():
    facts={'facts':[{'fact_id':'F0','kind':'object','value':{'class':'building','attributes':['red']},'source_images':['a']},{'fact_id':'F1','kind':'object','value':{'class':'road','attributes':['narrow']},'source_images':['a']}]}
    class OneAtATime:
        def generate(self,prompt):
            evidence=json.loads(prompt.split('Evidence: ',1)[1]); f=evidence[0]
            return {'captions':[{'text':'A visible '+f['value']['class']+' with stable features.','source_fact_ids':[f['fact_id']]}]}
    rows=generate_captions(OneAtATime(),facts,{'types':['partial'],'per_type':2})
    assert len(rows)==2 and rows[0]['source_fact_ids']!=rows[1]['source_fact_ids']
