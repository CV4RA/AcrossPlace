import copy,json
from text_generation.visual_parser import VisualFacts
from text_generation.caption_generator import Captions
from text_generation.prompts import VISUAL,CAPTION,prompt_facts

TASK_VERSION='vllm-place-batch-v2'
TYPES=['detailed','concise','partial','relational']
INSTRUCTIONS={
 'detailed':'Describe several distinctive static features, without annotation commentary. Bind camera directions to one view.',
 'concise':'One short sentence per description about distinctive object appearance. No directions, counts or spatial relationships unless explicitly present in evidence. Use different supported details.',
 'partial':'Each sentence describes only ONE object and a subset of its visible attributes. Prefer a different provided object for the second description. No directions, counts or relationships.',
 'relational':'Describe a verified spatial relationship from one camera. Explicitly say from this viewpoint for directions, and set the provided view_id. Cite relation and needed object observations.'}
LIMITS={'detailed':(18,55),'concise':(6,22),'partial':(5,16),'relational':(12,38)}

def visual_task():
    schema=VisualFacts.model_json_schema(by_alias=False)
    schema['properties']['static_objects']['minItems']=1; schema['properties']['static_objects']['maxItems']=6
    schema['properties']['spatial_relations']['maxItems']=6
    ids=['o'+str(i) for i in range(1,7)]
    obj=schema['$defs']['StaticObject']; obj['properties']['object_id']={'type':'string','enum':ids}; obj['required']=list(obj['properties'])
    for endpoint in ['subject','object']: schema['$defs']['Relation']['properties'][endpoint]={'type':'string','enum':ids}
    prompt=VISUAL+'\nUse object IDs o1, o2, ... in order, at most o6. A relation may ONLY reference IDs you actually declared. If a road or another endpoint is not among the six objects, omit that relation. Never reference an undeclared object.'
    return prompt,schema

def caption_task(fused,kind,c):
    count=c.get('descriptions_per_type',2)
    facts=[f for f in fused['facts'] if not f.get('conflicting') and not f.get('uncertain')]
    if not facts: raise ValueError('No grounded facts')
    anon=prompt_facts(facts); paths=sorted({p for f in facts for p in f['source_images']}); views={f'view_{i}':p for i,p in enumerate(paths)}
    selected=facts
    if kind in ['concise','partial']:
        patterns=[f for f in facts if f['kind']=='object']
        if patterns: selected=patterns[:max(2,count)]
    if kind=='relational':
        relations=[f for f in facts if f['kind']=='relation']
        if not relations: raise ValueError('No verified spatial relations for relational text')
        chosen=relations[0]['value']['view']; selected=[f for f in facts if f['value'].get('view')==chosen]
    ids=sorted({f['fact_id'] for f in selected}); evidence=[f for f in anon if f['fact_id'] in ids]
    limits=c.get('lengths',{}).get(kind,LIMITS[kind])
    prompt=CAPTION.format(count=count,kind=kind,language=c.get('language','en'),type_instruction=INSTRUCTIONS[kind],min_words=limits[0],max_words=limits[1],facts=json.dumps(evidence,separators=(',',':')))
    prompt=prompt.replace('\nEvidence: ','\nOutput exactly '+str(count)+' captions. claims is optional; omit it to avoid repeating the descriptions. Allowed source_fact_ids: '+json.dumps(ids)+'.\nEvidence: ')
    schema=Captions.model_json_schema(by_alias=False); array=schema['properties']['captions']; array.update(minItems=count,maxItems=count)
    row=schema['$defs']['Caption']; row['properties'].pop('claims',None); row['required']=[k for k in row.get('required',[]) if k!='claims']; row['additionalProperties']=False
    row['properties']['source_fact_ids']['items']={'type':'string','enum':ids}
    row['properties']['source_fact_ids']['maxItems']=min(12,len(ids))
    allowed_views=sorted({a['value']['view'] for a in evidence if a['value'].get('view')})
    if kind=='relational': row['properties']['view_id']={'type':'string','enum':allowed_views}; row['required']=sorted(set(row['required']+['view_id']))
    elif kind in ['partial','concise']: row['properties']['view_id']={'type':'null'}
    else: row['properties']['view_id']={'anyOf':[{'type':'null'},{'type':'string','enum':list(views)}]}
    return prompt,schema,ids,views

def validate_visual(value,image_id):
    facts=VisualFacts.model_validate(value).model_dump()
    if not facts['static_objects']: raise ValueError('No static evidence')
    facts['image_id']=image_id; return facts

def validate_captions(value,kind,ids,views,count):
    captions=Captions.model_validate(value).captions
    if len(captions)!=count: raise ValueError('Wrong caption count')
    seen=set(); result=[]
    for caption in captions:
        if not set(caption.source_fact_ids)<=set(ids): raise ValueError('Unknown source facts')
        if caption.view_id is not None and caption.view_id not in views: raise ValueError('Unknown camera view')
        normalized=' '.join(caption.text.lower().split())
        if normalized in seen: raise ValueError('Repeated identical caption; choose different grounded features')
        seen.add(normalized); result.append(dict(caption.model_dump(),type=kind,content=caption.text,bound_view=views.get(caption.view_id)))
    return result
