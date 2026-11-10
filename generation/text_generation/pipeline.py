"""Staged, resumable place-aware annotation. Production CLI never substitutes labels.
A factory may be injected by unit tests only; reports identify such test doubles.
"""
import json,time,copy
from pathlib import Path
from collections import Counter
from utils.io import read_jsonl,write_jsonl,digest
from utils.generation_cache import GenerationCache,image_identity
from utils.vlm_loader import VLM
from utils.seed import seed_all
from data.retrieval_protocols import plan_protocol
from text_generation.visual_parser import parse_visual,parse_visual_batch
from text_generation.place_fusion import fuse_place_facts
from text_generation.caption_generator import generate_captions
from text_generation.quality_filter import filter_caption,fact_frequencies,quality_report
from text_generation.prompts import PROMPT_VERSION

class PlaceTextPipeline:
    def __init__(self,c,resume=None,limit_places=None,vlm_factory=None):
        self.c=copy.deepcopy(c); self.out=Path(c['output']['directory']); self.annotations=self.out/'annotations'; self.reports=self.out/'reports'; self.cache_dir=self.out/'cache'
        for p in [self.annotations,self.reports,self.cache_dir]: p.mkdir(parents=True,exist_ok=True)
        self.resume=c.get('processing',{}).get('resume',True) if resume is None else resume
        self.limit=limit_places; self._vlm=None; self.factory=vlm_factory or VLM; self.test_double=vlm_factory is not None
        self.cache=GenerationCache(self.cache_dir/'generation.sqlite3',c.get('processing',{}).get('cache_enabled',True))
        self.failures=[]; self._signature=self.signature(); self.model_info=None
        self.run={'total_places':0,'total_images':0,'processed_images':0,'newly_inferred_images':0,'cached_images':0,'generated_texts':0,'valid_texts':0,'failed_samples':0,'average_seconds_per_image':None,'average_seconds_per_place':None,'peak_gpu_memory_bytes':None,'model_inference_executed':False,'test_double':self.test_double}
        (self.reports/'generation_config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    def signature(self):
        model=self.c['model']; name=model['name']; identity={'requested_revision':model.get('revision','main')}
        path=Path(name)
        if path.is_dir():
            identity['files']=[(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(path.iterdir()) if p.is_file()]
        elif not (path.is_absolute() or name.startswith(('.', '~')) or ':' in name):
            try:
                from huggingface_hub import try_to_load_from_cache
                cached=try_to_load_from_cache(name,'config.json',revision=model.get('revision','main'),cache_dir=model.get('cache_dir','outputs/models_cache'))
                if isinstance(cached,str): identity['resolved_config']=str(Path(cached).resolve())
            except ImportError: identity['cache_lookup']='unavailable'
        return digest([model,identity,self.c['generation'],PROMPT_VERSION,'visual-schema-v2'])
    @property
    def vlm(self):
        if self._vlm is None:
            self._vlm=self.factory(self.c['model'],self.c['generation']); self.model_info=getattr(self._vlm,'info',{'model':self.c['model']['name'],'revision':'test-double'})
        return self._vlm
    def audit_failure(self,stage,sample,e):
        record={'prompt_version':PROMPT_VERSION,'generation_signature':self._signature,'stage':stage,'sample':sample,'error':str(e),'status':'failed','responses':getattr(self._vlm,'last_responses',[])}
        self.failures.append(record)
        with (self.reports/'failure_events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
    def finish(self,stage):
        self.run['failed_samples']=len(self.failures)
        if self._vlm is not None and not self.test_double:
            self.run['model_inference_executed']=bool(getattr(self._vlm,'last_responses',[])); self.run['peak_gpu_memory_bytes']=self._vlm.monitor.peak_bytes if self._vlm.monitor.enabled else None
            self.run['model_info']=self.model_info
            write_jsonl(self.reports/(stage+'_gpu_log.jsonl'),self._vlm.monitor.records)
        existing={}
        path=self.reports/'generation_metrics.json'
        if path.exists(): existing=json.loads(path.read_text(encoding='utf-8'))
        if existing.get('config_signature')!=self._signature: existing={'config_signature':self._signature}
        existing[stage]=dict(self.run)
        extract=existing.get('extract',{}); generate=existing.get('generate',{}); filtered=existing.get('filter',{})
        stages=[v for k,v in existing.items() if k in ['extract','generate','filter']]
        peaks=[v['peak_gpu_memory_bytes'] for v in stages if v.get('peak_gpu_memory_bytes') is not None]
        existing['summary']={'total_places':extract.get('total_places',self.run['total_places']),'total_images':extract.get('total_images',self.run['total_images']),'processed_images':extract.get('processed_images',0),'generated_texts':generate.get('generated_texts',0),'valid_texts':filtered.get('valid_texts',0),'retained_texts':filtered.get('retained_texts',0),'failed_samples':sum(v.get('failed_samples',0) for v in stages),'average_seconds_per_image':extract.get('average_seconds_per_image'),'average_seconds_per_place_extract':extract.get('average_seconds_per_place'),'average_seconds_per_place_caption':generate.get('average_seconds_per_place'),'peak_gpu_memory_bytes':max(peaks) if peaks else None,'model_inference_executed':any(v.get('model_inference_executed',False) for v in stages),'test_double':self.test_double}
        path.write_text(json.dumps(existing,indent=2),encoding='utf-8')
        write_jsonl(self.reports/(stage+'_failures.jsonl'),self.failures)
        print(json.dumps({'stage':stage,**self.run},ensure_ascii=False))
    def prepare(self):
        dc=self.c['dataset']; source=Path(dc.get('index','outputs/data/places.jsonl'))
        if not source.is_file(): raise FileNotFoundError(f'Prepare the official dataset first: {source}; run scripts/prepare_gsv.py')
        rows=read_jsonl(source); count=self.limit if self.limit is not None else dc.get('max_places')
        if self.limit is not None and self.limit>len(rows): raise ValueError(f'Requested {self.limit} places but index has {len(rows)}; rerun prepare_gsv.py --limit-places {self.limit}')
        rows=rows[:count]
        if dc.get('place_ids') is not None: rows=[r for r in rows if r['place_id'] in dc['place_ids']]
        pc=dict(self.c.get('protocol',{}),seed=dc.get('random_seed',42),max_source_views=dc.get('max_views_per_place',4))
        plans,report=plan_protocol(rows,pc); self.run.update(total_places=len(rows),total_images=sum(len(r['images']) for r in rows))
        write_jsonl(self.cache_dir/'protocol_plan.jsonl',plans); (self.reports/'protocol_plan_report.json').write_text(json.dumps(report,indent=2))
        (self.reports/'protocol_plan_signature.json').write_text(json.dumps({'signature':digest([pc,rows]),'source_images_only_sent_to_vlm':True,'eligible_places':len(plans)}))
        return plans
    def extract(self,dry_run=False):
        plans=self.prepare()
        if dry_run:
            self.run['dry_run']=True; self.finish('dry_run'); return plans
        image_records=[]; fused_rows=[]; image_elapsed=[]; place_elapsed=[]
        def failed_image(im,error):
            row={'image_id':im['image_id'],'place_id':place['place_id'],'split':place['split'],'path':im['path'],'status':'failed','facts':None,'error':str(error),'model':self.model_info,'prompt_version':PROMPT_VERSION,'generation':self.c['generation']}
            image_records.append(row)
            with (self.annotations/'image_facts.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        write_jsonl(self.annotations/'image_facts.jsonl',[])
        for plan in plans:
            place=plan['place']; started=time.perf_counter(); inferred_before=self.run['newly_inferred_images']; visuals={}; pending=[]
            for im in plan['source_images']:
                try:
                    identity=image_identity(im['path']); key=digest(['image',identity,self._signature])
                    cached=self.cache.get(key) if self.resume else None
                    if cached is not None:
                        cached=dict(cached,place_id=place['place_id'],split=place['split'],image_id=im['image_id'],path=im['path'])
                        visuals[im['path']]=cached['facts']; image_records.append(dict(cached,cache_hit=True)); self.run['cached_images']+=1
                        with (self.annotations/'image_facts.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(dict(cached,cache_hit=True))+'\n')
                    else: pending.append((im,key,identity))
                except (OSError,ValueError) as e: self.audit_failure('image_identity',im,e); failed_image(im,e)
            bs=self.c['model'].get('batch_size',1)
            if bs<1: raise ValueError('model.batch_size must be positive')
            for begin in range(0,len(pending),bs):
                batch=pending[begin:begin+bs]; vlm=self.vlm; tick=time.perf_counter()
                seed_all(int(digest([self.c['dataset'].get('random_seed',42),[im['image_id'] for im,_,_ in batch]])[:8],16))
                try:
                    if len(batch)==1: values=[parse_visual(vlm,batch[0][0]['path'],self.c['generation'].get('max_retries',3),batch[0][0]['image_id'])]
                    else: values=parse_visual_batch(vlm,[im['path'] for im,_,_ in batch],self.c['generation'].get('max_retries',3))
                    successful=list(zip(batch,values))
                except (ValueError,TypeError,OSError,RuntimeError) as e:
                    # A bad image must not discard other images in its batch.
                    successful=[]
                    if len(batch)==1: self.audit_failure('visual',batch[0][0],e); failed_image(batch[0][0],e)
                    else:
                        for im,key,identity in batch:
                            try: successful.append(((im,key,identity),parse_visual(vlm,im['path'],self.c['generation'].get('max_retries',3),im['image_id'])))
                            except (ValueError,TypeError,OSError,RuntimeError) as error: self.audit_failure('visual',im,error); failed_image(im,error)
                    if hasattr(vlm,'monitor'): vlm.monitor.clear()
                elapsed=time.perf_counter()-tick
                for (im,key,identity),facts in successful:
                    facts['image_id']=im['image_id']; image_elapsed.append(elapsed/len(batch)); self.run['newly_inferred_images']+=1
                    row={'image_id':im['image_id'],'place_id':place['place_id'],'split':place['split'],'path':im['path'],'image_identity':identity,'facts':facts,'status':'success','model':self.model_info,'prompt_version':PROMPT_VERSION,'generation':self.c['generation'],'cache_signature':self._signature,'seconds':None if self.test_double else elapsed/len(batch),'timing_scope':'batch elapsed divided by batch size, including retries','responses':getattr(vlm,'last_responses',[]) if self.c.get('processing',{}).get('save_intermediate',True) else []}
                    self.cache.put(key,row); image_records.append(row); visuals[im['path']]=facts
                    print(json.dumps({'event':'image_done','place_id':place['place_id'],'image_id':im['image_id'],'seconds':row['seconds'],'cache_hit':False}),flush=True)
                    with (self.annotations/'image_facts.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if visuals:
                fusion=fuse_place_facts(visuals,plan['source_images'],self.c.get('fusion',{}))
                fused_rows.append({'place_id':place['place_id'],'place':place,'fusion':fusion,'facts':fusion['facts'],'source_image_paths':[im['path'] for im in plan['source_images']],'target_image_paths':[im['path'] for im in plan['gallery_images']],'protocol':plan['protocol'],'generation_signature':self._signature,'planning_config_signature':digest([self.c.get('protocol',{}),self.c['dataset'].get('max_views_per_place',4),self.c['dataset'].get('random_seed',42)])})
                with (self.annotations/'place_facts.events.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(fused_rows[-1],ensure_ascii=False)+'\n')
            else: self.audit_failure('fusion',place['place_id'],'No successful source views')
            if self.run['newly_inferred_images']>inferred_before: place_elapsed.append(time.perf_counter()-started)
        write_jsonl(self.annotations/'image_facts.jsonl',image_records); write_jsonl(self.annotations/'place_facts.jsonl',fused_rows)
        self.run['processed_images']=sum(row['status']=='success' for row in image_records)
        if not self.test_double:
            self.run['average_seconds_per_image']=sum(image_elapsed)/len(image_elapsed) if image_elapsed else None
            self.run['average_seconds_per_place']=sum(place_elapsed)/len(place_elapsed) if place_elapsed else None
        self.finish('extract'); return fused_rows
    def generate(self):
        path=self.annotations/'place_facts.jsonl'
        if not path.exists(): self.extract()
        fused=read_jsonl(path)
        requested=self.limit if self.limit is not None else self.c['dataset'].get('max_places')
        fused=fused[:requested]; rows=[]; generation_times=[]
        self.run['total_places']=len(fused); self.run['total_images']=sum(len(r['place']['images']) for r in fused)
        write_jsonl(self.annotations/'place_texts_raw.jsonl',[])
        for r in fused:
            if r.get('planning_config_signature')!=digest([self.c.get('protocol',{}),self.c['dataset'].get('max_views_per_place',4),self.c['dataset'].get('random_seed',42)]): raise ValueError('Source/gallery planning config changed; rerun extraction')
            if r['generation_signature']!=self._signature: raise ValueError('Extraction model/prompt/generation config changed; rerun extract_visual_facts.py')
            place=r['place']; seed=self.c['dataset'].get('random_seed',42); seed_all(int(digest([seed,place['place_id']])[:8],16))
            key=digest(['captions',r['fusion'],self._signature]); cached=self.cache.get(key) if self.resume else None
            try:
                if cached is not None: row={**cached,**{k:v for k,v in place.items() if k!='texts'},'source_image_paths':r['source_image_paths'],'target_image_paths':r['target_image_paths'],'protocol':r['protocol'],'fusion':r['fusion']}
                else:
                    vlm=self.vlm; tick=time.perf_counter()
                    captions=generate_captions(vlm,r['fusion'],self.c['generation'],self.cache if self.resume else None,key); generation_times.append(time.perf_counter()-tick); texts=[]
                    lookup={f['fact_id']:f for f in r['facts']}; image_ids={im['path']:im['image_id'] for im in place['images']}
                    for i,t in enumerate(captions):
                        sources=sorted({p for fid in t['source_fact_ids'] for p in lookup[fid]['source_images']})
                        t.update(text_id=place['place_id']+f'_T{i}',source_images=sources,source_image_ids=[image_ids[p] for p in sources],quality_score=None,generation={'model':self.model_info,'prompt_version':PROMPT_VERSION,'seed':seed,'parameters':self.c['generation']})
                        texts.append(t)
                    row=dict(place,texts=texts,source_image_paths=r['source_image_paths'],target_image_paths=r['target_image_paths'],protocol=r['protocol'],fusion=r['fusion'],generation_signature=self._signature)
                    self.cache.put(key,row)
                rows.append(row)
                print(json.dumps({'event':'place_texts_done','place_id':place['place_id'],'texts':len(row['texts']),'cache_hit':cached is not None}),flush=True)
                with (self.annotations/'place_texts_raw.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
            except (ValueError,TypeError,RuntimeError) as e: self.audit_failure('caption',place['place_id'],e)
        write_jsonl(self.annotations/'place_texts_raw.jsonl',rows); self.run['generated_texts']=sum(len(r['texts']) for r in rows)
        if generation_times and not self.test_double: self.run['average_seconds_per_place']=sum(generation_times)/len(generation_times)
        self.finish('generate'); return rows
    def filter(self):
        path=self.annotations/'place_texts_raw.jsonl'
        if not path.exists(): raise FileNotFoundError('Run generate_place_texts.py before filtering')
        rows=read_jsonl(path); frequencies={s:fact_frequencies([r['fusion'] for r in rows if r['split']==s]) for s in ['train','val','test']}; audited=[]; filtered=[]; split_counts=Counter(r['split'] for r in rows)
        vlm=self.vlm if self.c.get('quality',{}).get('vlm_verification',False) else None
        for r in rows:
            texts=[]; seen=set()
            for t in r['texts']:
                texts.append(filter_caption(t,r['fusion'],r,vlm,self.c.get('quality',{}),seen,frequencies[r['split']],split_counts[r['split']]))
            audited.append(dict(r,texts=texts)); filtered.append(dict(r,texts=[t for t in texts if t['accepted']]))
        write_jsonl(self.annotations/'place_texts_raw.jsonl',audited); write_jsonl(self.annotations/'place_texts_filtered.jsonl',filtered)
        compatibility=self.c['output'].get('compatibility_annotations')
        if compatibility: write_jsonl(compatibility,filtered)
        metrics_path=self.reports/'generation_metrics.json'
        metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        latest_failures=sum(metrics.get(s,{}).get('failed_samples',0) for s in ['extract','generate'])
        report=quality_report(audited,latest_failures); report.update(test_double=self.test_double,quality_config=self.c.get('quality',{}),failed_samples_scope='latest extraction and generation stages')
        (self.reports/'text_quality_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        self.run.update(total_places=len(rows),total_images=sum(len(r['images']) for r in rows),generated_texts=report['generated_texts'],valid_texts=report['valid_texts'],retained_texts=report['retained_texts']); self.finish('filter'); return report
    def close(self):
        self.cache.close()
        if self._vlm is not None:
            monitor=getattr(self._vlm,'monitor',None); self._vlm=None
            if monitor: monitor.clear()
