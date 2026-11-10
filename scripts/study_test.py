"""Freeze all validation-complete study jobs, then test without selection changes."""
import argparse
from pathlib import Path
import torch
from place_retrieval.data import records, verify_formal
from place_retrieval.encoding import encode_split
from place_retrieval.evaluation import checkpoint_state, test as evaluate_test
from place_retrieval.io import digest, load_config, read_json, run_lock, save_torch, sha256, write_json
from place_retrieval.metrics import coverage

def freeze(root):
    root=Path(root);plan=read_json(root/'plan.json');files={str(root/'plan.json'):sha256(root/'plan.json')}
    for job in plan['jobs']:
        c=load_config(job['config']);out=Path(job['output']);done=read_json(out/'job_complete.json')
        if digest(c)!=job['config_digest'] or done['config_digest']!=job['config_digest']:
            raise ValueError('Job configuration/completion changed')
        paths=[Path(job['config']),out/'job_complete.json',out/'coarse/best.pt']
        if job['action']!='zero_coarse':paths.extend([out/'evaluation/selection.json',out/'evaluation/val_result.json'])
        else:paths.append(out/'coarse/val_result.json')
        for path in paths:files[str(path)]=sha256(path)
    frozen={'plan_fingerprint':plan['fingerprint'],'files':files,'test_hyperparameter_search':False}
    dest=root/'test_freeze/global.json'
    if dest.exists() and read_json(dest)!=frozen:raise ValueError('A different global freeze exists')
    write_json(dest,frozen);return frozen

def verify_gate(root):
    root=Path(root);gate=read_json(root/'test_freeze/global.json');plan=read_json(root/'plan.json')
    if plan['fingerprint']!=gate['plan_fingerprint']:raise ValueError('Frozen plan identity differs')
    for path,expected in gate['files'].items():
        if sha256(path)!=expected:raise ValueError(f'Frozen artifact changed: {path}')
    return plan

def coarse_test(c):
    manifest=verify_formal(c['data']);out=Path(c['output'])/'evaluation';dest=out/'test_result.json'
    if dest.exists():return
    device=torch.device(c['device'])
    model,processor,heads,epoch=checkpoint_state(Path(c['output'])/'coarse/best.pt',c,manifest,device)
    rows=records(c['data'],'test');features=encode_split(model,processor,heads,rows,c,device)
    save_torch(out/'coarse_test_features.pt',{'features':features,'records':rows,'formal_fingerprint':manifest['fingerprint']})
    write_json(dest,{'split':'test','mode':'coarse_only','hyperparameter_search_on_test':False,'coarse_only':coverage(features,rows),'checkpoint_epoch':epoch,'formal_fingerprint':manifest['fingerprint']})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--freeze',action='store_true');group.add_argument('--role',choices=('comparison','ablation','auxiliary'))
    args=p.parse_args()
    if args.freeze:
        with run_lock(args.root/'test_freeze'):freeze(args.root)
        print('All declared validation choices frozen. No test has run by this command.')
    else:
        with run_lock(args.root/'test_workers'/args.role):
            plan=verify_gate(args.root)
            for j in plan['jobs']:
                if j['role']!=args.role:continue
                verify_gate(args.root);c=load_config(j['config']);torch.set_num_threads(c.get('cpu_threads',8))
                if j['action']=='zero_coarse':coarse_test(c)
                else:evaluate_test(c,Path(j['output'])/'evaluation/selection.json')
