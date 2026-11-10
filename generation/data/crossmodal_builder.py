import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from data.retrieval_protocols import plan_protocol
from utils.io import read_jsonl,write_jsonl

def validate_spatial_splits(rows,buffer_m=25):
    if buffer_m<=0: return
    ll=[]; splits=[]
    for row in rows:
        for gps in row.get('gps',[]): ll.append(gps); splits.append(row['split'])
    if not ll: return
    ll=np.radians(ll); xyz=np.stack([np.cos(ll[:,0])*np.cos(ll[:,1]),np.cos(ll[:,0])*np.sin(ll[:,1]),np.sin(ll[:,0])],1)
    for a,b in cKDTree(xyz).query_pairs(2*np.sin(buffer_m/6371000/2),output_type='ndarray'):
        if splits[a]!=splits[b]: raise ValueError('GPS buffer split leakage detected')

def build_crossmodal_dataset(c):
    rows=read_jsonl(c['annotations']); validate_spatial_splits(rows,c.get('spatial_buffer_m',25))
    registry={}
    for row in rows:
        for im in row['images']:
            for key in [('path',str(Path(im['path']).resolve())),('panoid',im['panoid']),('image_id',im['image_id'])]:
                if key in registry and registry[key]!=row['split']: raise ValueError('Image/panorama split leakage')
                registry[key]=row['split']
    # Keep EXACT generation-side membership. Never choose a new source set here.
    if c.get('protocol','B') not in ['A','B']: raise ValueError('Protocol must be A/B')
    exported=[]; skipped=[]
    from eval.protocols import dhash,HashTree
    tree=HashTree(); source_paths=set(); source_panos=set()
    for row in rows:
        source_paths.update(row.get('source_image_paths',[]))
        for im in row['images']:
            if im['path'] in row.get('source_image_paths',[]): source_panos.add(im['panoid'])
    if c.get('protocol','B')=='B':
        for p in source_paths: tree.add(int.from_bytes(dhash(p),'big'))
    all_tids=set(); all_iids={}; seen_places=set()
    for row in rows:
        if row['place_id'] in seen_places: raise ValueError('Duplicate place IDs')
        seen_places.add(row['place_id'])
        texts=[t for t in row['texts'] if t.get('accepted',True) and (c.get('include_uncertain',True) or t.get('quality_status')!='uncertain')]
        if not texts: skipped.append({'place_id':row['place_id'],'reason':'no_retained_texts'}); continue
        declared=set(row.get('source_image_paths',[])); provenance={p for t in texts for p in t.get('source_images',[])}
        if not provenance<=declared: raise ValueError('Caption evidence outside generation-side views')
        image_ids={im['path']:im['image_id'] for im in row['images']}
        if not declared<=set(image_ids): raise ValueError('Unknown generation source image')
        for t in texts:
            if set(t.get('source_image_ids',[]))!={image_ids[p] for p in t.get('source_images',[])}: raise ValueError('Caption provenance IDs do not match source paths')
        paths=declared if c.get('protocol','B')=='A' else set(row.get('target_image_paths',[]))
        if c.get('protocol','B')=='B' and paths&source_paths: raise ValueError('Gallery includes a caption source image')
        images=[]
        for im in row['images']:
            if im['path'] not in paths: continue
            if c.get('protocol','B')=='B' and (im['panoid'] in source_panos or tree.near(int.from_bytes(dhash(im['path']),'big'),c.get('duplicate_hamming',4))): continue
            iid=im['image_id']
            if iid in all_iids and all_iids[iid]!=im['path']: raise ValueError('Image ID collision')
            all_iids[iid]=im['path']; images.append(im)
        if not images: skipped.append({'place_id':row['place_id'],'reason':'no_independent_images'}); continue
        for t in texts:
            if t['text_id'] in all_tids: raise ValueError('Duplicate text IDs')
            all_tids.add(t['text_id'])
        exported.append(dict(row,texts=texts,evaluation_images=images))
    out=Path(c.get('output','outputs')); annotation=out/'annotations'; annotation.mkdir(parents=True,exist_ok=True)
    t2i_q=[]; t2i_g=[]; i2t_q=[]; i2t_g=[]
    for r in exported:
        common={'place_id':r['place_id'],'positive_place_ids':[r['place_id']],'split':r['split'],'protocol':c.get('protocol','B'),'dataset':r['dataset']}
        for t in r['texts']:
            obj=dict(t,**common)
            if c.get('protocol','B')=='A': obj['positive_image_ids']=t['source_image_ids']
            t2i_q.append(obj); i2t_g.append(obj)
        for im in r['evaluation_images']:
            obj=dict(im,**common)
            if c.get('protocol','B')=='A': obj['positive_text_ids']=[t['text_id'] for t in r['texts'] if im['image_id'] in t['source_image_ids']]
            t2i_g.append(obj); i2t_q.append(obj)
    for name,records in [('t2i_queries',t2i_q),('t2i_gallery',t2i_g),('i2t_queries',i2t_q),('i2t_gallery',i2t_g)]: write_jsonl(annotation/(name+'.jsonl'),records)
    split_dir=out/'splits'; split_dir.mkdir(parents=True,exist_ok=True)
    for split in ['train','val','test']: (split_dir/(split+'_places.json')).write_text(json.dumps([r['place_id'] for r in exported if r['split']==split],indent=2))
    report={'protocol':c.get('protocol','B'),'input_places':len(rows),'exported_places':len(exported),'query_texts':len(t2i_q),'gallery_images':len(t2i_g),'query_images':len(i2t_q),'gallery_texts':len(i2t_g),'skipped':skipped,'split_counts':dict((s,sum(r['split']==s for r in exported)) for s in ['train','val','test']),'i2t_pairing':'direct_source' if c.get('protocol','B')=='A' else 'independent_view','gallery_pixels_used_for':'duplicate audit only; never visual facts or language generation'}
    report_dir=out/'reports'; report_dir.mkdir(parents=True,exist_ok=True); (report_dir/'crossmodal_report.json').write_text(json.dumps(report,indent=2))
    return report
