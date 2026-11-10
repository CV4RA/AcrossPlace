"""Plan source/gallery membership before any VLM or fusion sees image content.
Gallery pixels are read ONLY by the perceptual-duplicate audit, never by the VLM.
"""
import random
from pathlib import Path
from eval.protocols import dhash,HashTree
from utils.io import digest

def plan_protocol(places,config=None):
    c=config or {}; protocol=c.get('protocol','B')
    if protocol not in ['A','B']: raise ValueError('Generation protocols support A (direct) and B (independent views)')
    rng=random.Random(c.get('seed',42)); plans=[]; rejected=[]
    seen=set(); paths={}; panos={}
    for place in places:
        pid=place['place_id']; split=place['split']
        if split not in ['train','val','test']: raise ValueError('Unknown split')
        if pid in seen: raise ValueError('Duplicate place ID or place split leakage')
        seen.add(pid)
        images=place['images']
        for image in images:
            image.setdefault('image_id','I_'+digest(str(Path(image['path']).resolve()))[:24])
            for registry,key in [(paths,str(Path(image['path']).resolve())),(panos,image['panoid'])]:
                if key in registry and registry[key]!=split: raise ValueError('Image/panorama split leakage')
                registry[key]=split
        groups={}
        for im in images: groups.setdefault(im['panoid'],[]).append(im)
        names=sorted(groups); rng.shuffle(names)
        if protocol=='B' and len(names)<2: rejected.append({'place_id':pid,'reason':'insufficient_independent_panoramas'}); continue
        n=min(c.get('source_panorama_groups',2),len(names)-(1 if protocol=='B' else 0))
        chosen=set(names[:max(1,n)]); source=[im for im in images if im['panoid'] in chosen][:c.get('max_source_views',4)]
        if protocol=='A': targets=source
        else: targets=[im for im in images if im['panoid'] not in chosen]
        plans.append({'place':place,'place_id':pid,'split':split,'protocol':protocol,'source_images':source,'gallery_images':targets})
    tree=HashTree(); global_sources={im['path'] for p in plans for im in p['source_images']}; global_panos={im['panoid'] for p in plans for im in p['source_images']}
    bad_sources=set()
    if protocol=='B':
        for path in sorted(global_sources):
            try: tree.add(int.from_bytes(dhash(path),'big'))
            except OSError as e: bad_sources.add(path); rejected.append({'path':path,'reason':'unreadable_source:'+str(e)})
    result=[]; removed=0
    for plan in plans:
        source=[im for im in plan['source_images'] if im['path'] not in bad_sources]
        gallery=[]
        for im in plan['gallery_images']:
            try:
                if protocol=='B' and (im['path'] in global_sources or im['panoid'] in global_panos or tree.near(int.from_bytes(dhash(im['path']),'big'),c.get('duplicate_hamming',4))):
                    removed+=1; rejected.append({'place_id':plan['place_id'],'image_id':im['image_id'],'reason':'source_or_near_duplicate_gallery'}); continue
                gallery.append(im)
            except OSError as e: rejected.append({'place_id':plan['place_id'],'image_id':im['image_id'],'reason':'unreadable_gallery:'+str(e)})
        if not source or not gallery: rejected.append({'place_id':plan['place_id'],'reason':'no_valid_source_or_independent_gallery'}); continue
        result.append(dict(plan,source_images=source,gallery_images=gallery))
    return result,{'protocol':protocol,'input_places':len(places),'eligible_places':len(result),'near_duplicate_gallery_removed':removed,'issues':rejected,'duplicate_detection':'64-bit dHash heuristic; not a guarantee of semantic independence','seed':c.get('seed',42)}
