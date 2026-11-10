"""Descriptive retrieval stratification by independent human labels; no tuning."""
import json
import os
from pathlib import Path
import statistics
import torch
import numpy as np
from benchmark import read, write, sha, immutable, lines, model_config, retrieval_hits, metrics
from place_retrieval.encoding import encode_split
from place_retrieval.evaluation import checkpoint_state
from place_retrieval.io import save_torch, seed_all, run_lock
from place_retrieval.models import load_siglip

ROOT=Path(os.environ.get('ACROSSPLACE_DIAGNOSTICS', 'runs/diagnostics'))


def replacement_hits(scores, positive, index):
    """Replace the focus place's caption in a full, one-caption-per-place gallery."""
    rank=1+int((scores>positive).sum())-int(scores[index]>positive)
    rank+=int(((scores==positive)&(np.arange(len(scores))<index)).sum())
    return [float(rank<=k) for k in [1,5,10]]


def main():
    out=ROOT/'human_retrieval';out.mkdir(exist_ok=True)
    plan=read(ROOT/'plan.json');annotations=read(ROOT/'inputs/accepted_review.json')['rows']
    models=[m for m in plan['models'] if not m.get('protocol') or m['protocol']=='P1']
    identity={'plan_sha256':sha(ROOT/'plan.json'),'human_sha256':sha(ROOT/'inputs/accepted_review.json'),
              'code_sha256':sha(__file__),'models':[m['id'] for m in models],
              'T2I':'Full original P1 test target-image gallery; same-place positives',
              'I2T':'Diagnostic one-caption-per-place gallery; replace focus-place caption per human item; average both target-image queries',
              'label_scope':'Visibility refers to either displayed target image; I2T results average individual target views',
              'selection_on_human_labels':False}
    immutable(out/'freeze.json',identity)
    places=sorted([p for p in lines(Path(plan['protocols']['P1']['formal'])/'valid.jsonl') if p['split']=='test'],key=lambda p:p['place_id'])
    lookup={p['place_id']:p for p in places};place_index={p['place_id']:i for i,p in enumerate(places)}
    assert all(r['place_id'] in lookup for r in annotations)
    spec={p['place_id']:p for p in read(ROOT/'specs/P1.json')['pairs']}
    images=[];texts={};background=[]
    for p in places:
        fields={'place_id':p['place_id'],'split':'test'}
        images.extend(dict(i,**fields) for i in p['images'] if i['path'] in p['target_image_paths'])
        text=dict(spec[p['place_id']]['text'] if p['place_id'] in spec else sorted(p['texts'],key=lambda t:t['text_id'])[0],**fields)
        texts[text['text_id']]=text;background.append(text)
    for r in annotations:
        texts[r['text_id']]={'text_id':r['text_id'],'text':r['text'],'place_id':r['place_id'],'split':'test'}
    rows={'image':images,'text':[texts[k] for k in sorted(texts)]}
    torch.set_num_threads(8);seed_all(42)
    for model in models:
        write(out/'status.json',{'stage':'running','model':model['id'],'pid':os.getpid()})
        dest=out/(model['id']+'.json')
        if dest.exists():continue
        c=model_config(model,plan);device=torch.device('cuda')
        cache=out/(model['id']+'_features.pt')
        if cache.exists():
            saved=torch.load(cache,map_location='cpu',weights_only=True)
            assert saved['identity']==identity
            features=saved['features']
        else:
            if model.get('checkpoint'):
                assert sha(model['checkpoint'])==read(model['selection'])['coarse_sha256']
                backbone,processor,heads,_=checkpoint_state(model['checkpoint'],c,read(Path(plan['protocols']['P1']['formal'])/'manifest.json'),device)
            else:backbone,processor,heads=load_siglip(c['models']['siglip'],c['training'],device)
            features=encode_split(backbone,processor,heads,rows,c,device)
            save_torch(cache,{'identity':identity,'features':features})
            del backbone,processor,heads;torch.cuda.empty_cache()
        f={k:v.numpy() for k,v in features.items()}
        ti={r['text_id']:i for i,r in enumerate(rows['text'])}
        ii={r['path']:i for i,r in enumerate(images)}
        query=f['text'][[ti[r['text_id']] for r in annotations]]
        t2i=retrieval_hits(query,f['image'],[r['place_id'] for r in annotations],[r['place_id'] for r in images])
        bg=f['text'][[ti[r['text_id']] for r in background]]
        per_item=[]
        for j,r in enumerate(annotations):
            query_images=f['image'][[ii[p] for p in r['target_images']]]
            background_scores=query_images@bg.T
            positive=query_images@query[j]
            item_hits=[replacement_hits(s,p,place_index[r['place_id']]) for s,p in zip(background_scores,positive)]
            i2t=np.asarray(item_hits).mean(0).tolist()
            per_item.append({'audit_id':r['audit_id'],'text_id':r['text_id'],'place_id':r['place_id'],
                             'T2I':t2i[j],'I2T_controlled':i2t})
        groups={}
        for field in ['source_correctness','target_visibility','place_specificity','rule_status']:
            groups[field]={}
            for label in sorted({r[field] for r in annotations}):
                indexes=[i for i,r in enumerate(annotations) if r[field]==label]
                groups[field][label]={'n':len(indexes),**{d:{f'Recall@{k}':statistics.mean(per_item[i][d][a] for i in indexes) for a,k in enumerate([1,5,10])} for d in ['T2I','I2T_controlled']}}
        write(dest,{'identity':identity,'model':model['id'],'training_seed':model.get('training_seed'),
                    'image_gallery':len(images),'controlled_text_gallery_places':len(background),
                    'groups':groups,'per_item':per_item,'feature_sha256':sha(cache)})
        print(json.dumps({'model_complete':model['id'],'human_items':len(per_item)}),flush=True)
    write(out/'completion.json',{'status':'complete','identity':identity,'models':len(models),
                                 'interpretation':'Descriptive subgroup associations, single reviewer; no causal quality effect or population prevalence claim'})
    write(out/'status.json',{'stage':'complete','models':len(models)})


if __name__=='__main__':
    with run_lock(ROOT/'human_retrieval'):
        main()
