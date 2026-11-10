"""Cross-place concurrent annotation with durable per-image/per-type checkpoints."""
import json,time,copy,sqlite3
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from collections import Counter
from utils.io import read_jsonl,write_jsonl,digest
from utils.vllm_client import VLLMClient
from utils.generation_cache import GenerationCache
from data.retrieval_protocols import plan_protocol
from text_generation.prompts import PROMPT_VERSION
from text_generation.batch_tasks import TASK_VERSION,TYPES,visual_task,caption_task,validate_visual,validate_captions
from text_generation.place_fusion import fuse_place_facts
from text_generation.quality_filter import fact_frequencies,filter_caption,quality_report

class ConcurrentPlacePipeline:
    def __init__(self,c,client=None):
        self.c=c; self.out=Path(c['output']['directory']); self.ann=self.out/'annotations'; self.reports=self.out/'reports'
        for p in [self.ann,self.reports,self.out/'cache']: p.mkdir(parents=True,exist_ok=True)
        self.cache=GenerationCache(self.out/'cache'/'vllm.sqlite3'); self.cache.db.execute('PRAGMA journal_mode=WAL'); self.cache.db.execute('PRAGMA synchronous=NORMAL')
        self.client=client or VLLMClient(c['vllm']); self.stats={}; self.failures=[]
        self.signature=digest([c['model_identity'],c['generation'],c.get('fusion',{}),PROMPT_VERSION,TASK_VERSION])
        (self.reports/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    def log(self,value):
        print(json.dumps(value,ensure_ascii=False),flush=True)
        with (self.reports/'events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(value,ensure_ascii=False)+'\n')
    def execute(self,stage,tasks,validator):
        workers=self.c['vllm'].get('concurrency',64); attempts=self.c['generation'].get('max_retries',2); started=time.perf_counter()
        stats={'stage':stage,'cached':0,'succeeded':0,'failed':0,'requests':0,'completion_tokens':0,'prompt_tokens':0}; last=time.monotonic()
        def request(task,attempt,error):
            revised=dict(task); revised['max_tokens']=min(self.c['generation'].get('retry_max_tokens',1536),task['max_tokens']+attempt*256)
            if error: revised['prompt']+='\nThe previous response failed validation: '+str(error)[:1000]+'. Correct the complete JSON using only the supplied evidence.'
            return self.client.complete(revised)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending={}; queue=iter(tasks); exhausted=False
            while pending or not exhausted:
                while not exhausted and len(pending)<workers*2:
                    try: task=next(queue)
                    except StopIteration: exhausted=True; break
                    if self.cache.get(task['key']) is not None: stats['cached']+=1; continue
                    future=pool.submit(request,task,0,None); pending[future]=(task,0); stats['requests']+=1
                if not pending: continue
                done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for future in done:
                    task,attempt=pending.pop(future)
                    response=None
                    try:
                        response=future.result(); value=validator(response['value'],task)
                        record={'value':value,'response':response,'task_id':task['key'],'place_id':task['place_id'],'kind':task.get('kind'),'signature':self.signature,'attempts':attempt+1}
                        self.cache.put(task['key'],record); stats['succeeded']+=1
                        for key in ['completion_tokens','prompt_tokens']: stats[key]+=response['usage'].get(key,0)
                    except Exception as error:
                        if attempt<attempts:
                            pending[pool.submit(request,task,attempt+1,error)]=(task,attempt+1); stats['requests']+=1
                        else:
                            failure={'stage':stage,'task_id':task['key'],'place_id':task['place_id'],'kind':task.get('kind'),'error':str(error),'attempts':attempt+1,'response':response}; self.failures.append(failure); self.cache.put(task['key'],failure,status='failed'); stats['failed']+=1; self.log({'event':'task_failed',**failure})
                if time.monotonic()-last>=10:
                    elapsed=time.perf_counter()-started; update=dict(stats,event='progress',elapsed_s=elapsed,inflight=len(pending),requests_per_second=stats['succeeded']/elapsed)
                    (self.reports/'progress.json').write_text(json.dumps(update,indent=2)); self.log(update); last=time.monotonic()
        stats['elapsed_s']=time.perf_counter()-started; stats['requests_per_second']=stats['succeeded']/max(.001,stats['elapsed_s']); self.stats[stage]=stats; self.log(stats)
        (self.reports/'metrics.json').write_text(json.dumps(self.stats,indent=2)); write_jsonl(self.reports/'failures.jsonl',self.failures)
    def prepare(self,limit=None):
        rows=read_jsonl(self.c['dataset']['index'])
        if limit is not None: rows=rows[:limit]
        else: rows=rows[:self.c['dataset'].get('max_places')]
        frozen=self.c['dataset'].get('plan_file')
        if frozen:
            wanted={r['place_id'] for r in rows}; current={r['place_id']:r for r in rows}
            plans=[p for p in read_jsonl(frozen) if p['place_id'] in wanted]
            for plan in plans:
                row=current[plan['place_id']]
                if plan['split']!=row['split'] or {i['path'] for i in plan['place']['images']}!={i['path'] for i in row['images']}: raise ValueError('Frozen plan differs from current place metadata')
                if {i['path'] for i in plan['source_images']}&{i['path'] for i in plan['gallery_images']}: raise ValueError('Frozen plan has source/gallery overlap')
            report={'protocol':self.c['protocol'].get('protocol','B'),'input_places':len(rows),'eligible_places':len(plans),'frozen_plan':str(frozen),'issues':[]}
        else: plans,report=plan_protocol(rows,self.c['protocol'])
        write_jsonl(self.out/'cache'/'plans.jsonl',plans)
        (self.reports/'protocol_report.json').write_text(json.dumps(report,indent=2)); self.log({'event':'protocol_ready',**{k:v for k,v in report.items() if k!='issues'}}); return plans
    def image_jobs(self,plans):
        prompt,schema=visual_task()
        for p in plans:
            for image in p['source_images']:
                key=digest(['visual',self.signature,image['image_id'],image.get('sha256',image['path'])])
                yield {'key':key,'seed':int(key[:8],16),'place_id':p['place_id'],'image_id':image['image_id'],'image':image['path'],'prompt':prompt,'schema':schema,'max_tokens':self.c['generation'].get('visual_max_tokens',1024)}
    def extract(self,plans):
        jobs=list(self.image_jobs(plans)); self.execute('visual',jobs,lambda value,task:validate_visual(value,task['image_id']))
        by_place={}; image_rows=[]
        for task in jobs:
            record=self.cache.get(task['key'])
            if record:
                facts=record['value']; by_place.setdefault(task['place_id'],{})[task['image']]=facts
                image_rows.append({'image_id':task['image_id'],'place_id':task['place_id'],'path':task['image'],'facts':facts,'status':'success','generation':{'signature':self.signature,'model':self.c['model_identity'],'prompt_version':PROMPT_VERSION,'engine':'vllm'},'inference':{k:v for k,v in record['response'].items() if k not in ['value']}})
        write_jsonl(self.ann/'image_facts.jsonl',image_rows); fused=[]
        for plan in plans:
            visuals=by_place.get(plan['place_id'],{})
            if not visuals: continue
            fusion=fuse_place_facts(visuals,plan['source_images'],self.c.get('fusion',{})); fused.append({'place':plan['place'],'place_id':plan['place_id'],'split':plan['split'],'fusion':fusion,'source_image_paths':[i['path'] for i in plan['source_images']],'target_image_paths':[i['path'] for i in plan['gallery_images']],'protocol':plan['protocol']})
        write_jsonl(self.ann/'place_facts.jsonl',fused); return fused
    def caption_jobs(self,fused):
        for row in fused:
            for kind in self.c['generation'].get('types',TYPES):
                try: prompt,schema,ids,views=caption_task(row['fusion'],kind,self.c['generation'])
                except ValueError as error:
                    self.failures.append({'stage':'caption_preparation','place_id':row['place_id'],'kind':kind,'error':str(error)}); continue
                key=digest(['caption',self.signature,row['place_id'],kind,row['fusion'],prompt])
                yield {'key':key,'seed':int(key[:8],16),'place_id':row['place_id'],'kind':kind,'prompt':prompt,'schema':schema,'ids':ids,'views':views,'max_tokens':self.c['generation'].get('caption_max_tokens',640)}
    def generate(self,fused):
        jobs=list(self.caption_jobs(fused)); count=self.c['generation'].get('descriptions_per_type',2)
        self.execute('caption',jobs,lambda value,task:validate_captions(value,task['kind'],task['ids'],task['views'],count))
        texts={}
        for task in jobs:
            record=self.cache.get(task['key'])
            if record: texts.setdefault(task['place_id'],[]).extend(record['value'])
        raw=[]
        for row in fused:
            place=row['place']; candidates=texts.get(row['place_id'],[]); facts={f['fact_id']:f for f in row['fusion']['facts']}; ids={im['path']:im['image_id'] for im in place['images']}
            for index,t in enumerate(candidates):
                paths=sorted({p for fid in t['source_fact_ids'] for p in facts[fid]['source_images']})
                t.update(text_id=place['place_id']+'_T'+str(index),source_images=paths,source_image_ids=[ids[p] for p in paths],quality_score=None,generation={'signature':self.signature,'model':self.c['model_identity'],'prompt_version':PROMPT_VERSION,'task_version':TASK_VERSION,'engine':'vllm'})
            raw.append(dict(place,texts=candidates,source_image_paths=row['source_image_paths'],target_image_paths=row['target_image_paths'],fusion=row['fusion'],protocol=row['protocol'],missing_types=[k for k in TYPES if k not in {t['type'] for t in candidates}]))
        write_jsonl(self.ann/'place_texts_raw.jsonl',raw); return raw
    def filter(self,raw):
        frequencies={s:fact_frequencies([r['fusion'] for r in raw if r['split']==s]) for s in ['train','val','test']}; counts=Counter(r['split'] for r in raw); audited=[]; filtered=[]
        for row in raw:
            seen=set(); texts=[filter_caption(t,row['fusion'],row,None,self.c.get('quality',{}),seen,frequencies[row['split']],counts[row['split']]) for t in row['texts']]
            audited.append(dict(row,texts=texts)); filtered.append(dict(row,texts=[t for t in texts if t['accepted']]))
        write_jsonl(self.ann/'place_texts_raw.jsonl',audited); write_jsonl(self.ann/'place_texts_filtered.jsonl',filtered)
        report=quality_report(audited,len(self.failures)); report.update(model=self.c['model_identity'],signature=self.signature,engine='vllm',missing_type_places=sum(bool(r['missing_types']) for r in raw)); (self.reports/'text_quality_report.json').write_text(json.dumps(report,indent=2)); return report
    def close(self): self.cache.close()
