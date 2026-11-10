import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from utils.io import config,write_jsonl
from data.gsv_cities import parse_gsv
from data.place_index import split_places

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/dataset.yaml'); ap.add_argument('--limit-places',type=int); a=ap.parse_args(); c=config(a.config)
    fixed=c.get('place_ids')
    if c.get('place_list'): fixed=[s.strip() for s in Path(c['place_list']).read_text(encoding='utf-8').splitlines() if s.strip()]
    errors=[]; out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
    try:
        rows=parse_gsv(c['root'],c.get('cities'),a.limit_places or c.get('limit_places'),c.get('images_per_place'),place_ids=fixed,seed=c.get('seed',42),shuffle=c.get('shuffle',False),skip_bad_images=c.get('skip_bad_images',True),verify_images=c.get('verify_images',True),errors=errors)
    finally: write_jsonl(out/'image_errors.jsonl',errors)
    rows,report=split_places(rows,**c.get('split',{})); report['skipped_images']=len(errors)
    write_jsonl(out/'places.jsonl',rows)
    splits=Path(c.get('splits_output','outputs/splits')); splits.mkdir(parents=True,exist_ok=True)
    for split in ['train','val','test']:
        subset=[r for r in rows if r['split']==split]; write_jsonl(out/(split+'.jsonl'),subset)
        (splits/(split+'_places.json')).write_text(json.dumps([r['place_id'] for r in subset],indent=2),encoding='utf-8')
    (out/'split_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
