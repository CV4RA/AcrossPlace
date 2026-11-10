import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time
from utils.io import config,read_jsonl
from text_generation.vllm_pipeline import ConcurrentPlacePipeline
from data.crossmodal_builder import build_crossmodal_dataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/vllm_15k.yaml'); ap.add_argument('--limit-places',type=int); ap.add_argument('--stage',choices=['all','prepare','visual','caption','filter'],default='all'); a=ap.parse_args(); c=config(a.config); p=ConcurrentPlacePipeline(c); started=time.time()
    try:
        if a.stage in ['all','prepare','visual']: plans=p.prepare(a.limit_places)
        if a.stage=='prepare': return
        if a.stage in ['all','visual']: fused=p.extract(plans)
        else: fused=read_jsonl(p.ann/'place_facts.jsonl')
        if a.stage=='visual': return
        if a.stage in ['all','caption']: raw=p.generate(fused)
        else: raw=read_jsonl(p.ann/'place_texts_raw.jsonl')
        report=p.filter(raw); retrieval=build_crossmodal_dataset({'annotations':str(p.ann/'place_texts_filtered.jsonl'),'output':str(p.out),'protocol':c['protocol'].get('protocol','B'),'duplicate_hamming':c['protocol'].get('duplicate_hamming',4),'spatial_buffer_m':25})
        summary={'status':'complete_with_failures' if p.failures else 'complete','elapsed_s':time.time()-started,'quality':report,'retrieval':retrieval,'metrics':p.stats}; (p.reports/'completion.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
        if p.failures: raise SystemExit(2)
    finally: p.close()
if __name__=='__main__': main()
