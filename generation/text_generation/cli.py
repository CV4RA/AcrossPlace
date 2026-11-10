import argparse
from utils.io import config
from text_generation.pipeline import PlaceTextPipeline

def run(stage):
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/text_generation.yaml'); ap.add_argument('--limit-places',type=int)
    resume=ap.add_mutually_exclusive_group(); resume.add_argument('--resume',action='store_true',default=None); resume.add_argument('--no-resume',action='store_false',dest='resume')
    ap.add_argument('--dry-run',action='store_true'); ap.add_argument('--filter',action='store_true',help='Run rules-only quality filtering after generation')
    a=ap.parse_args(); pipeline=PlaceTextPipeline(config(a.config),a.resume,a.limit_places)
    try:
        if a.dry_run: pipeline.extract(dry_run=True)
        elif stage=='extract': pipeline.extract()
        elif stage=='generate':
            pipeline.generate()
            if a.filter: pipeline.filter()
        elif stage=='filter': pipeline.filter()
        else: raise ValueError('Unknown stage')
        if pipeline.failures: raise SystemExit(f'Completed with {len(pipeline.failures)} failed samples; inspect outputs/reports and resume')
    except Exception as e:
        pipeline.audit_failure(stage,'stage_failure',e); pipeline.finish('error'); raise
    finally: pipeline.close()
