"""Create a portable, newly hashed diagnostic plan without changing ranking logic."""
import argparse
import json
from pathlib import Path
import shutil
import sys
from benchmark import sha,write,read

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','p0-formal','p1-formal','p0-runs','p1-runs','image-root','model-root']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();repo=Path(__file__).resolve().parents[1]
    if (root/'plan.json').exists():raise FileExistsError('Diagnostic plan already exists; use a new output directory')
    root.mkdir(parents=True,exist_ok=True)
    # Only reference code and its fresh source inventory, not caches or credentials.
    for folder in ['src','scripts','configs']:
        shutil.copytree(repo/folder,root/'code'/folder,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copy2(repo/'pyproject.toml',root/'code/pyproject.toml')
    source_files={p.relative_to(root/'code').as_posix():sha(p) for p in (root/'code').rglob('*') if p.is_file()}
    write(root/'code/release_files.json',source_files)
    historical=repo/'results/historical_source_review.json';dest=root/'inputs/historical_human_review.json';write(dest,read(historical))
    canonical='/root/autodl-tmp/datasets/GSVCities'
    human=read(repo/'results/human_review.json')
    for r in human['rows']:
        for key in ['source_images','target_images']:r[key]=[canonical+'/'+s.split('GSVCities/',1)[-1] for s in r[key]]
    write(root/'inputs/accepted_review.json',human)
    template=read(Path(__file__).with_name('plan.archive.json'))
    plan={k:v for k,v in template.items() if k not in ('input_files','models','protocols')}
    plan.update(root=str(root),image_root=str(a.image_root.resolve()),canonical_prefix=canonical,
                uncertain_file=str(a.p0_formal.resolve()/'valid_plus_uncertain_same_places.jsonl'),
                human_review=str(dest),archive_plan_sha256=sha(Path(__file__).with_name('plan.archive.json')))
    plan['protocols']={}
    for name,formal in [('P0',a.p0_formal),('P1',a.p1_formal)]:
        plan['protocols'][name]={'formal':str(formal.resolve()),'fingerprint':template['protocols'][name]['fingerprint']}
        if read(formal/'manifest.json')['fingerprint']!=plan['protocols'][name]['fingerprint']:raise ValueError(f'{name} identity differs')
    plan['models']=[];files={dest,Path(plan['uncertain_file'])}
    for formal in [a.p0_formal,a.p1_formal]:files.add(formal.resolve()/'manifest.json')
    for m in template['models']:
        m=dict(m)
        if m.get('checkpoint'):
            runs=a.p0_runs if m['protocol']=='P0' else a.p1_runs
            base=runs.resolve()/f"T4_siglip_place_s{m['training_seed']}"
            m.update(config=str(base/'coarse/config.json'),checkpoint=str(base/'coarse/best.pt'),selection=str(base/'evaluation/selection.json'),completion=str(base/'coarse/completion.json'))
            files.update(Path(m[k]) for k in ['config','checkpoint','selection','completion'])
        else:
            m['directory']=str(a.model_root.resolve()/('blip-itm-base-coco' if m['kind']=='blip' else 'siglip2-base-patch16-384'))
            m['config']=str(a.p1_runs.resolve()/'T4_siglip_place_s42/coarse/config.json');files.add(Path(m['config']))
        plan['models'].append(m)
    for directory in ['siglip2-base-patch16-384','blip-itm-base-coco']:
        files.update(p for p in (a.model_root.resolve()/directory).iterdir() if p.is_file() and p.suffix in ('.json','.safetensors','.model','.txt'))
    plan['input_files']={str(p.resolve()):sha(p) for p in sorted(files)}
    write(root/'plan.json',plan);print(json.dumps({'plan':str(root/'plan.json'),'input_files':len(files)}))

if __name__=='__main__':main()
