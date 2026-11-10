"""Fuse evidence patterns, not unverified cross-view physical object instances."""
import json

def fuse_place_facts(image_facts,view_metadata=None,config=None):
    c=config or {}; facts=[]; patterns=[]; transients=[]; conflicts=[]; uncertain=[]
    threshold=c.get('attribute_similarity_threshold',.85); min_conf=c.get('min_confidence',.5)
    if not 0<=threshold<=1: raise ValueError('attribute_similarity_threshold must be in [0,1]')
    def add(kind,value,paths,**kwargs):
        fact={'fact_id':f'F{len(facts)}','kind':kind,'value':value,'source_images':sorted(set(paths)),'support':len(set(paths))/max(1,len(image_facts)),**kwargs}; facts.append(fact); return fact
    scenes={}
    for path,v in image_facts.items():
        scenes.setdefault(v['scene_type'],[]).append(path)
        transients.append({'source_image':path,'objects':v.get('transient_objects',[]),'environment':v.get('environment',{})})
        uncertain.extend({'source_image':path,'observation':u} for u in v.get('uncertain_facts',[]))
        objects=v['static_objects']; by_id={o.get('object_id',f'obj_{i:03d}'):o for i,o in enumerate(objects)}
        for i,obj in enumerate(objects):
            category=obj.get('category',obj.get('class')); attrs=set(a.lower().replace(' ','_') for a in obj['attributes']); oid=obj.get('object_id',f'obj_{i:03d}')
            confidence=obj.get('confidence'); weak=confidence is not None and confidence<min_conf
            observation=add('view_object',{'class':category,'attributes':sorted(attrs),'relative_position':obj.get('relative_position','unknown'),'view':path},[path],source_observations=[{'image':path,'object_id':oid}],confidence=confidence,uncertain=weak,instance_identity='within_view_only')
            if weak: uncertain.append({'source_image':path,'fact_id':observation['fact_id'],'reason':'low_model_self_confidence'}); continue
            matched=None
            for pattern in patterns:
                union=attrs|pattern['attributes']; sim=len(attrs&pattern['attributes'])/len(union) if union else 1.
                if category==pattern['category'] and sim>=threshold: matched=pattern; break
            if matched is None:
                matched={'category':category,'attributes':attrs,'paths':[],'observations':[]}; patterns.append(matched)
            else: matched['attributes'] &= attrs
            matched['paths'].append(path); matched['observations'].append({'image':path,'object_id':oid})
        rels=[]
        for relation in v.get('spatial_relations',[]):
            # Resolve local object IDs into visible categories and attributes for captions.
            value=dict(relation,view=path)
            for endpoint in ['subject','object']:
                obj=by_id.get(relation[endpoint])
                if obj: value[endpoint+'_description']={'class':obj.get('category',obj.get('class')),'attributes':obj['attributes']}
            rels.append(add('relation',value,[path],coordinate_frame='camera',conflicting=False))
        opposite={'left_of':'right_of','right_of':'left_of','in_front_of':'behind','behind':'in_front_of','above':'below','below':'above'}
        for i,a in enumerate(rels):
            for b in rels[i+1:]:
                av,bv=a['value'],b['value']
                if av['subject']==bv['subject'] and av['object']==bv['object'] and opposite.get(av['predicate'])==bv['predicate']:
                    a['conflicting']=b['conflicting']=True; conflicts.append({'source_image':path,'fact_ids':[a['fact_id'],b['fact_id']],'reason':'opposing_relations_within_same_camera_frame'})
    for scene,paths in sorted(scenes.items()): add('scene',{'scene_type':scene},paths)
    for p in patterns:
        add('object',{'class':p['category'],'attributes':sorted(p['attributes'])},p['paths'],source_observations=p['observations'],instance_identity='unresolved_shared_appearance_pattern')
    stable=[f['fact_id'] for f in facts if f['kind'] in ['object','scene'] and len(f['source_images'])>=c.get('min_shared_views',2) and f['support']>=c.get('min_support',.5)]
    return {'facts':facts,'stable_landmarks':stable,'view_specific_observations':[f['fact_id'] for f in facts if f['kind']=='view_object'],'transient_information':transients,'conflicts':conflicts,'uncertain_observations':uncertain,'views':view_metadata or [],'direction_frame':'source_view_only','cross_view_instance_matching':'unresolved; only appearance patterns are grouped','merge_config':c}

def fuse_visual_facts(image_facts,view_metadata=None): return fuse_place_facts(image_facts,view_metadata)
