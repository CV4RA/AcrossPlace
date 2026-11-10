import json
from pydantic import BaseModel,Field,AliasChoices
from text_generation.prompts import CAPTION,prompt_facts
from utils.io import digest

class Caption(BaseModel):
    text:str=Field(min_length=1,validation_alias=AliasChoices('text','content'))
    source_fact_ids:list[str]=Field(min_length=1)
    claims:list[str]=Field(default_factory=list)
    view_id:str|None=None
class Captions(BaseModel):
    captions:list[Caption]

def generate_captions(vlm,fused,c,cache=None,cache_key=None):
    result=[]
    facts=[f for f in fused['facts'] if not f.get('conflicting') and not f.get('uncertain')]
    if not facts: raise ValueError('No verified visual evidence for captions')
    anonymous=prompt_facts(facts); valid_ids={f['fact_id'] for f in facts}
    paths=sorted({p for f in facts for p in f['source_images']}); views={f'view_{i}':p for i,p in enumerate(paths)}
    instructions={
        'detailed':'Describe several distinctive visible static features in one or two sentences. Avoid directions unless needed and correctly bound to one camera. Do not include statements about identity or verification.',
        'concise':'Write ONE short sentence highlighting one or two distinctive landmarks. Do not use camera directions, counts, or explanations.',
        'partial':'Each description covers ONLY ONE of the supplied static objects and its attributes. Choose DIFFERENT objects or distinct supported feature subsets for different descriptions. One short sentence each. Do not summarize the whole scene. Do not use directions, counts or relationships.',
        'relational':'Emphasize ONE or TWO verified spatial relations from a SINGLE camera view. Explicitly frame any direction with from this viewpoint and set view_id. Cite the relation and relevant object observations.'}
    default_limits={'detailed':(18,55),'concise':(6,22),'partial':(5,16),'relational':(12,38)}
    for kind in c.get('types',['detailed','concise','partial','relational']):
        if kind not in ['detailed','concise','partial','relational']: raise ValueError('Unknown caption type')
        count=c.get('descriptions_per_type',c.get('per_type',2)); errors=[]
        selected=facts
        if kind in ['partial','concise']:
            patterns=[f for f in facts if f['kind']=='object']
            if patterns: selected=patterns[:max(2,count)] if kind=='partial' else patterns[:2]
        if kind=='relational':
            relations=[f for f in facts if f['kind']=='relation']
            if relations:
                chosen=relations[0]['value'].get('view')
                selected=[f for f in facts if f['value'].get('view')==chosen]
        selected_ids={f['fact_id'] for f in selected}; selected_anonymous=[f for f in anonymous if f['fact_id'] in selected_ids]
        limits=c.get('lengths',{}).get(kind,default_limits[kind])
        type_key=digest(['caption_type',cache_key,kind,sorted(selected_ids),limits])
        existing=cache.get(type_key) if cache is not None else None
        if existing is not None:
            result.extend(existing); continue
        rows=[]
        for attempt in range(c.get('max_retries',c.get('retries',3))+1):
            try:
                remaining=count-len(rows)
                call_evidence=selected_anonymous
                if rows and kind=='partial':
                    used={fid for row in rows for fid in row['source_fact_ids']}
                    alternative=[f for f in selected_anonymous if f['fact_id'] not in used]
                    if alternative: call_evidence=alternative
                call_ids={f['fact_id'] for f in call_evidence}
                prompt=CAPTION.format(count=remaining,kind=kind,language=c.get('language','en'),type_instruction=instructions[kind],min_words=limits[0],max_words=limits[1],facts=json.dumps(call_evidence))
                feedback=''
                if errors: feedback+='Correct this previous error: '+errors[-1]['error']+'. '
                if rows: feedback+='Already collected descriptions: '+json.dumps([r['text'] for r in rows])+'. Generate '+str(remaining)+' NEW descriptions; do not repeat these. '
                if feedback: prompt=prompt.replace('\nEvidence: ','\n'+feedback+'\nEvidence: ')
                captions=Captions.model_validate(vlm.generate(prompt)).captions
                if not captions: raise ValueError('No captions returned')
                for caption in captions:
                    if len(rows)>=count: break
                    if not set(caption.source_fact_ids)<=call_ids: raise ValueError('Unknown or excluded source facts')
                    if caption.view_id is not None and caption.view_id not in views: raise ValueError('Unknown observation viewpoint')
                    if any(' '.join(caption.text.lower().split())==' '.join(r['text'].lower().split()) for r in rows): continue
                    row=caption.model_dump(); row.update(type=kind,content=caption.text,bound_view=views.get(caption.view_id)); rows.append(row)
                if len(rows)==count:
                    if cache is not None: cache.put(type_key,rows)
                    result.extend(rows)
                    print(json.dumps({'event':'caption_type_done','type':kind,'texts':len(rows)}),flush=True)
                    break
                errors.append({'attempt':attempt,'error':f'Collected {len(rows)} valid descriptions; need {count-len(rows)} additional distinct descriptions'})
            except (ValueError,TypeError) as e: errors.append({'attempt':attempt,'error':str(e)})
        else: raise ValueError(f'Caption generation failed for {kind}: {errors}')
    return result
