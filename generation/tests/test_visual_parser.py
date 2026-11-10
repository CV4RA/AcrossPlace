import pytest
from utils.vlm_loader import clean_json
from text_generation.visual_parser import VisualFacts,parse_visual

def test_json_repair_does_not_change_strings():
    obj,repairs=clean_json('```json\n{"text":"a,}","items":[1,],}\n```')
    assert obj=={'text':'a,}','items':[1]}; assert repairs

def test_no_fake_json():
    with pytest.raises(ValueError): clean_json('not a structured response')

def test_reference_validation():
    v={'scene_type':'street','static_objects':[],'spatial_relations':[{'subject':'x','predicate':'beside','object':'y'}],'transient_objects':[],'environment':{},'uncertain_facts':[]}
    with pytest.raises(ValueError): VisualFacts.model_validate(v)

def test_retry_failure_is_explicit():
    class Failed:
        def generate(self,*args): return {'bad':1}
    with pytest.raises(ValueError,match='without fabricated'): parse_visual(Failed(),'x',retries=1)


def test_valid_json_with_empty_landmarks_is_not_a_success_label():
    class Empty:
        def generate(self,*args): return {'scene_type':'street','static_objects':[],'spatial_relations':[],'transient_objects':[],'environment':{},'uncertain_facts':[]}
    with pytest.raises(ValueError,match='No static evidence'): parse_visual(Empty(),'real-street',retries=0)
