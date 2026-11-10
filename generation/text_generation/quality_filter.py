"""Conservative, interpretable rule checks. Unresolved language is marked uncertain.
These scores are heuristics and must not be interpreted as human ground truth.
"""
import json,re,math
from collections import Counter
from difflib import SequenceMatcher
from text_generation.prompts import VERIFY,prompt_facts

QUALITY_VERSION='rules-v3'
OBJECTS={'building','road','intersection','bridge','sidewalk','tree','sign','traffic light','tower','church','mosque','temple','hospital','school','castle','stadium','airport','river','mountain','fountain','monument','statue','bench','wall','window','door','fence','railway','tunnel','car','bus','truck','pedestrian'}
ALIASES={'buildings':'building','roads':'road','trees':'tree','junction':'intersection','crossroads':'intersection','traffic lights':'traffic light','houses':'building','house':'building','windows':'window','cars':'car','people':'pedestrian','structures':'building','structure':'building','walls':'wall','roofing':'roof','roofs':'roof','adjacent to':'beside','next to':'beside','alongside':'beside','across from':'opposite','in back of':'behind'}
ATTRIBUTES={'red','blue','green','yellow','white','black','brown','gray','grey','brick','glass','wooden','circular','arched','triangular','stone','concrete','metal'}
STOP=set('camera perspective viewpoint captured located lies lying situated featuring features neighboring nearby another while within against backdrop set setting frames exterior context standing runs here facade facades has its both their s to the a an the is are this that and or of on in at to with from current viewing direction view stands stand near along has have surrounded by beside next left right front behind above below building road tree intersection multi story its it there visible place street scene shows showing features facade'.split())

def tokens(text):
    text=text.lower().replace('_',' ').replace('-',' ')
    for a,b in ALIASES.items(): text=re.sub(r'(?<!\w)'+re.escape(a)+r'(?!\w)',b,text)
    return re.findall(r'[a-z]+|\d+|[\u4e00-\u9fff]',text)

def fact_frequencies(fused_places):
    counts=Counter()
    for place in fused_places: counts.update(set(json.dumps(f['value'],sort_keys=True) for f in place['facts'] if f['kind'] in ['object','scene']))
    return counts

def filter_caption(caption,fused,place,vlm,c,seen,frequencies,n_places):
    text=caption.get('text',caption.get('content','')); normalized=' '.join(tokens(text)); reasons=[]; hard=[]; uncertain=[]
    def fail(reason): reasons.append(reason); hard.append(reason)
    if c.get('enable_duplication_check',True):
        for other in seen:
            a,b=set(normalized.split()),set(other.split()); overlap=len(a&b)/max(1,len(a|b))
            if normalized==other or (overlap>=c.get('duplicate_jaccard',.9) and SequenceMatcher(None,normalized,other).ratio()>=.88): fail('duplicate_or_near_duplicate'); break
    seen.add(normalized)
    unit_count=len(tokens(text)) if re.search(r'[\u4e00-\u9fff]',text) else len(text.split())
    if c.get('enable_length_check',True) and not c.get('min_words',4)<=unit_count<=c.get('max_words',100): fail('length')
    forbidden=[place['place_id'],place.get('metadata',{}).get('city','')]
    if any(s and re.search(r'(?<!\w)'+re.escape(s)+r'(?!\w)',text,re.I) for s in forbidden) or re.search(r'[-+]?\d{1,3}\.\d{3,}',text): fail('metadata_leak')
    by_id={f['fact_id']:f for f in fused['facts']}; ids=caption.get('source_fact_ids',[])
    evidence=[by_id[i] for i in ids if i in by_id]
    if not ids or len(evidence)!=len(ids): fail('invalid_evidence')
    if any(f.get('conflicting') or f.get('uncertain') for f in evidence): fail('conflicting_or_uncertain_source')
    evidence_words=set(tokens(json.dumps([f['value'] for f in evidence])))
    words=set(tokens(text)); lower=' '.join(tokens(text))
    if re.search(r'physical instances?|shared identity|identity (?:confirmed|matching)|confirm.*(?:instances?|identit)|no (?:other objects|shared)',text,re.I): fail('unsupported_meta_commentary')
    if c.get('enable_grounding_check',True) or c.get('enable_hallucination_check',True):
        unsupported=[]
        for term in OBJECTS|ATTRIBUTES:
            if set(tokens(term))<=words and not set(tokens(term))<=evidence_words: unsupported.append(term)
        if unsupported: fail('unsupported_objects_or_attributes:'+','.join(sorted(unsupported)))
        numbers=[x for x in words if x.isdigit() or x in {'one','two','three','four','five','six','seven','eight','nine','ten'}]
        # A count needs distinct observed object IDs in ONE camera, never a sum
        # of repeated appearances across unregistered viewpoints.
        observed={}
        for f in evidence:
            if f['kind']=='view_object':
                view=f['value'].get('view'); category=f['value'].get('class')
                observed.setdefault((view,category),set()).add(f['fact_id'])
        word_numbers=dict(zip(['one','two','three','four','five','six','seven','eight','nine','ten'],range(1,11)))
        for number in numbers:
            if number in evidence_words: continue
            amount=int(number) if number.isdigit() else word_numbers[number]
            count_supported=False
            for category in {f['value'].get('class') for f in evidence if f['kind']=='view_object'}:
                if category and re.search(r'\b'+re.escape(number)+r'\s+(?:[a-z]+\s+){0,4}'+re.escape(category)+r'\b',lower):
                    count_supported=any(cat==category and len(objects)>=amount for (view,cat),objects in observed.items())
            if not count_supported: fail('unsupported_count')
        unknown=words-evidence_words-STOP-set('urban narrow wide tall small large located exterior architecture architectural'.split())
        if unknown: uncertain.append('unresolved_language:'+','.join(sorted(unknown)))
    if c.get('enable_consistency_check',True):
        directions=words&{'left','right','front','behind','above','below'}
        relations=[f for f in evidence if f['kind']=='relation']
        view_objects=[f for f in evidence if f['kind']=='view_object']
        bound=caption.get('bound_view')
        for direction in directions:
            supported=[f for f in relations+view_objects if direction in tokens(json.dumps(f['value']))]
            if not bound or not any(f['value'].get('view')==bound for f in supported): fail('unverified_or_unbound_direction')
        if directions and not (words&{'view','viewing','viewpoint','camera'} or '视角' in text): fail('camera_frame_not_explicit')
        if caption.get('type')=='relational':
            if not relations: fail('relational_text_without_relation_evidence')
            if any(f['value'].get('view') and f['value']['view']!=bound for f in relations): fail('mixed_or_unbound_relation_views')
        relation_words={'beside','between','opposite','behind','above','below'}&words
        for term in relation_words:
            if not any(term in tokens(f['value']['predicate']) for f in relations): fail('unverified_spatial_relation:'+term)
    verdict=None
    if c.get('vlm_verification',False):
        if vlm is None: raise ValueError('vlm_verification requires a loaded VLM')
        errors=[]
        for attempt in range(c.get('max_retries',3)+1):
            try:
                verdict=vlm.generate(VERIFY+json.dumps({'description':text,'evidence':prompt_facts(evidence)}))
                if not all(isinstance(verdict.get(k),bool) for k in ['grounded','consistent']) or not isinstance(verdict.get('unsupported_claims'),list): raise ValueError('Malformed verifier JSON')
                break
            except (ValueError,TypeError) as e: errors.append(str(e))
        else: verdict={'grounded':False,'consistent':False,'unsupported_claims':[],'uncertain':True}; uncertain.append('vlm_verifier_failed:'+str(errors))
        if not verdict.get('uncertain') and (not verdict['grounded'] or verdict['unsupported_claims']): fail('vlm_unsupported_claims')
        if not verdict.get('uncertain') and not verdict['consistent']: fail('vlm_inconsistent')
        if verdict.get('uncertain'): uncertain.append('vlm_uncertain')
    rarity=[max(0.,math.log((n_places+1)/(frequencies.get(json.dumps(f['value'],sort_keys=True),0)+1))/max(1,math.log(n_places+1))) for f in evidence if f['kind'] in ['object','scene']]
    discrimination=sum(rarity)/len(rarity) if rarity else 0.
    if c.get('enable_discrimination_check',True) and discrimination<c.get('generic_threshold',.05): reasons.append('generic')
    grounding=0. if any(x.startswith(('unsupported','invalid','conflicting')) for x in hard) else (.5 if uncertain else 1.)
    consistency=0. if any('relation' in x or 'direction' in x or 'frame' in x for x in hard) else 1.
    weights=c.get('weights',[.5,.3,.2])
    if len(weights)!=3 or sum(weights)<=0: raise ValueError('quality.weights must have three entries with positive sum')
    score=sum(w*s for w,s in zip(weights,[grounding,consistency,discrimination]))/sum(weights)
    enabled=c.get('enabled',True)
    accepted=(not hard and (c.get('retain_uncertain',True) or not uncertain) and score>=c.get('threshold',0.)) if enabled else 'metadata_leak' not in hard
    reasons+=uncertain; status='rejected' if not accepted else ('uncertain' if uncertain else 'valid')
    sources=sorted({p for f in evidence for p in f['source_images']})
    return dict(caption,quality_rule_version=QUALITY_VERSION,text=text,content=text,quality_score=score,quality_components={'grounding':grounding,'consistency':consistency,'discrimination':discrimination},quality_is_heuristic=True,quality_status=status,filter_reasons=sorted(set(reasons)),accepted=accepted,source_images=sources,verification=verdict,difficulty='partial' if caption.get('type')=='partial' else ('low_discrimination' if 'generic' in reasons else 'standard'))

def quality_report(rows,failed_samples=0):
    texts=[t for r in rows for t in r['texts']]; reasons=Counter(x for t in texts for x in t['filter_reasons'])
    return {'quality_rule_version':QUALITY_VERSION,'heuristic_only':True,'total_places':len(rows),'generated_texts':len(texts),'valid_texts':sum(t['quality_status']=='valid' for t in texts),'retained_texts':sum(t['accepted'] for t in texts),'uncertain_texts':sum(t['quality_status']=='uncertain' for t in texts),'rejected_texts':sum(not t['accepted'] for t in texts),'failed_samples':failed_samples,'types':dict(Counter(t['type'] for t in texts)),'filter_reasons':dict(reasons),'difficulty':dict(Counter(t['difficulty'] for t in texts)),'mean_quality_score':sum(t['quality_score'] for t in texts)/len(texts) if texts else None}
