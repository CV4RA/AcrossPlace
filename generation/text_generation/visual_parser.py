from pydantic import BaseModel,Field,AliasChoices,model_validator,ConfigDict
from text_generation.prompts import VISUAL
from utils.io import digest
from utils.vlm_loader import VLM

class StaticObject(BaseModel):
    model_config=ConfigDict(extra='forbid')
    object_id:str=''
    category:str=Field(min_length=1,validation_alias=AliasChoices('category','class'))
    attributes:list[str]
    relative_position:str
    confidence:float|None=Field(default=None,ge=0,le=1)
class Relation(BaseModel):
    subject:str
    predicate:str
    object:str
class Environment(BaseModel):
    illumination:str='unknown'
    weather:str='unknown'
class VisualFacts(BaseModel):
    image_id:str=''
    scene_type:str=Field(min_length=1)
    static_objects:list[StaticObject]
    spatial_relations:list[Relation]
    transient_objects:list[str]
    environment:Environment
    uncertain_facts:list[str]
    @model_validator(mode='before')
    @classmethod
    def legacy_environment(cls,value):
        if isinstance(value,dict) and isinstance(value.get('environment'),list):
            value=dict(value); value['environment']={'illumination':'unknown','weather':'unknown'}
        return value
    @model_validator(mode='after')
    def references(self):
        for i,obj in enumerate(self.static_objects):
            if not obj.object_id: obj.object_id=f'obj_{i:03d}'
        ids=[o.object_id for o in self.static_objects]
        if len(ids)!=len(set(ids)): raise ValueError('Duplicate object_id')
        for relation in self.spatial_relations:
            for endpoint in ['subject','object']:
                value=getattr(relation,endpoint)
                if value in ids: continue
                matches=[o.object_id for o in self.static_objects if o.category==value]
                if len(matches)!=1: raise ValueError('Spatial relation references an unobserved or ambiguous object')
                setattr(relation,endpoint,matches[0])
        return self

def parse_visual(vlm,path,retries=3,image_id=None):
    errors=[]
    for attempt in range(retries+1):
        try:
            prompt=VISUAL if not errors else VISUAL+'\nPrevious attempt failed validation: '+errors[-1]['error']+'\nReturn a complete corrected JSON grounded in this image; reference only declared object IDs.'
            value=VisualFacts.model_validate(vlm.generate(prompt,path)).model_dump()
            if not value['static_objects']: raise ValueError('No static evidence extracted. Re-examine the image for visible permanent buildings, road, sidewalk or vegetation; do not return an empty template. If truly unrecognizable, explain uncertainty.')
            value['image_id']=image_id or 'I_'+digest(str(path))[:24]
            return value
        except (ValueError,TypeError) as e: errors.append({'attempt':attempt,'error':str(e)})
    raise ValueError(f'Visual parse failed, without fabricated labels: {errors}')

def parse_visual_batch(vlm,paths,retries=3):
    try:
        values=vlm.generate_batch([VISUAL]*len(paths),paths)
        if len(values)!=len(paths): raise ValueError('Wrong batch output length')
        result=[]
        for path,v in zip(paths,values):
            row=VisualFacts.model_validate(v).model_dump()
            if not row['static_objects']: raise ValueError('No static evidence extracted')
            row['image_id']='I_'+digest(str(path))[:24]; result.append(row)
        return result
    except (ValueError,TypeError): return [parse_visual(vlm,path,retries) for path in paths]
