"""Immutable place splits and exact source/gallery membership for all experiments."""
import copy,hashlib,json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from utils.io import read_jsonl,write_jsonl,digest
from data.crossmodal_builder import validate_spatial_splits
from eval.protocols import dhash,HashTree

FORMAL_VERSION='formal-valid-v1'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def compact(row,texts):
    keys=['place_id','dataset','split','gps','metadata','images','image_paths','source_image_paths','target_image_paths','protocol']
    result={k:copy.deepcopy(row[k]) for k in keys if k in row}
    result['texts']=[]
    for text in texts:
        item={k:copy.deepcopy(text[k]) for k in ['text_id','type','source_images','source_image_ids','source_fact_ids','bound_view','quality_status','accepted'] if k in text}
        item['text']=text.get('text',text.get('content')); item['content']=item['text']
        if not item['text']: raise ValueError('Empty text')
        result['texts'].append(item)
    return result

def prepare_formal(c):
    source=Path(c['annotations']); out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
    if c.get('prepared_only',False):
        manifest=json.loads((out/'manifest.json').read_text())
        if not c.get('expected_fingerprint') or manifest['fingerprint']!=c['expected_fingerprint']: raise ValueError('Migrated formal data fingerprint does not match the pinned configuration')
        for name,meta in manifest['files'].items():
            if file_sha(out/name)!=meta['sha256']: raise ValueError('Migrated frozen data file was modified: '+name)
        return manifest
    fingerprint=digest([FORMAL_VERSION,file_sha(source),c.get('spatial_buffer_m',25),c.get('duplicate_hamming',4),c.get('verify_near_duplicates',True)])
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['fingerprint']!=fingerprint: raise ValueError('Formal data is frozen; changed inputs require a new output directory')
        for name,meta in manifest['files'].items():
            if file_sha(out/name)!=meta['sha256']: raise ValueError('Frozen data file was modified: '+name)
        return manifest
    rows=read_jsonl(source)
    if len({r['place_id'] for r in rows})!=len(rows): raise ValueError('Duplicate place ID')
    validate_spatial_splits(rows,c.get('spatial_buffer_m',25))
    registry={}; tids=set(); global_sources=set(); global_panos=set()
    for row in rows:
        if row['split'] not in ['train','val','test']: raise ValueError('Unknown split')
        sources=set(row['source_image_paths']); targets=set(row['target_image_paths'])
        if sources&targets: raise ValueError('Source/gallery overlap')
        by_path={im['path']:im for im in row['images']}
        if not sources|targets<=set(by_path): raise ValueError('Unknown source/gallery path')
        global_sources.update(sources)
        global_panos.update(by_path[p]['panoid'] for p in sources)
        for im in row['images']:
            if not Path(im['path']).is_file(): raise FileNotFoundError(im['path'])
            im.setdefault('image_id','I_'+digest(im['path'])[:24])
            for key in [('path',im['path']),('panorama',im['panoid']),('image_id',im['image_id'])]:
                if key in registry and registry[key]!=row['split']: raise ValueError('Cross-split image/panorama leakage')
                registry[key]=row['split']
        for t in row['texts']:
            if t['text_id'] in tids: raise ValueError('Duplicate text ID')
            tids.add(t['text_id'])
            if not set(t['source_images'])<=sources: raise ValueError('Text evidence outside frozen source views')
            if t.get('source_image_ids') and set(t['source_image_ids'])!={by_path[p]['image_id'] for p in t['source_images']}: raise ValueError('Mismatched provenance IDs')
    targets={p for row in rows for p in row['target_image_paths']}
    if targets&global_sources: raise ValueError('Global source/gallery overlap')
    for row in rows:
        if any(im['panoid'] in global_panos for im in row['images'] if im['path'] in targets): raise ValueError('Global panorama source/gallery overlap')
    if c.get('verify_near_duplicates',True):
        tree=HashTree()
        with ThreadPoolExecutor(max_workers=8) as pool:
            for value in pool.map(dhash,sorted(global_sources)): tree.add(int.from_bytes(value,'big'))
            for path,value in zip(sorted(targets),pool.map(dhash,sorted(targets))):
                if tree.near(int.from_bytes(value,'big'),c.get('duplicate_hamming',4)): raise ValueError('Frozen gallery contains near-duplicate source image: '+path)
    strict=[]; expanded=[]; skipped=[]
    for row in rows:
        retained=[t for t in row['texts'] if t.get('accepted',True)]
        valid=[t for t in retained if t.get('quality_status')=='valid']
        if not valid or not row['target_image_paths']:
            skipped.append({'place_id':row['place_id'],'split':row['split'],'reason':'no_valid_texts_or_gallery'}); continue
        strict.append(compact(row,valid))
        # Future quality ablation uses exactly the same places; evaluation stays valid-only.
        expanded.append(compact(row,retained if row['split']=='train' else valid))
    files={}
    def save(name,records):
        write_jsonl(out/name,records); files[name]={'rows':len(records),'sha256':file_sha(out/name)}
    save('valid.jsonl',strict); save('valid_plus_uncertain_same_places.jsonl',expanded)
    for split in ['train','val','test']:
        subset=[r for r in strict if r['split']==split]; save(split+'.jsonl',subset)
        qs=[]; ims=[]
        for row in subset:
            common={'place_id':row['place_id'],'split':split,'positive_place_ids':[row['place_id']],'protocol':'B'}
            qs.extend(dict(t,**common) for t in row['texts'])
            ims.extend(dict(im,**common) for im in row['images'] if im['path'] in row['target_image_paths'])
        for name,records in [('t2i_queries',qs),('t2i_gallery',ims),('i2t_queries',ims),('i2t_gallery',qs)]: save(split+'_'+name+'.jsonl',records)
    save('skipped_places.jsonl',skipped)
    manifest={'version':FORMAL_VERSION,'fingerprint':fingerprint,'source_annotation':str(source),'source_sha256':file_sha(source),'quality_policy':'rule-valid only; not human ground truth','protocol':'B','splits':{s:{'places':sum(r['split']==s for r in strict),'texts':sum(len(r['texts']) for r in strict if r['split']==s),'gallery_images':sum(len(r['target_image_paths']) for r in strict if r['split']==s)} for s in ['train','val','test']},'valid_types':dict(Counter(t['type'] for r in strict for t in r['texts'])),'future_quality_ablation':'same places, unchanged valid-only val/test','near_duplicate_audit':bool(c.get('verify_near_duplicates',True)),'files':files}
    manifest_path.write_text(json.dumps(manifest,indent=2)); return manifest
