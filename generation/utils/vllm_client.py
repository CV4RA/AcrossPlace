"""Bounded OpenAI-compatible vLLM requests; inference runs in one GPU server."""
import threading,time,json
import requests
from utils.vlm_loader import clean_json

class VLLMClient:
    def __init__(self,c): self.c=c; self.local=threading.local()
    def complete(self,task):
        if not hasattr(self.local,'session'):
            self.local.session=requests.Session(); self.local.session.trust_env=False
        content=[{'type':'text','text':task['prompt']}]
        if task.get('image'): content.append({'type':'image_url','image_url':{'url':'file://'+task['image']}})
        payload={'model':self.c.get('served_model','qwen3-vl-4b'),'messages':[{'role':'user','content':content}],'temperature':0.,'top_p':1.,'seed':task['seed'],'max_tokens':task['max_tokens'],'response_format':{'type':'json_schema','json_schema':{'name':'place_annotation','schema':task['schema'],'strict':True}}}
        start=time.perf_counter()
        response=self.local.session.post(self.c['base_url'].rstrip('/')+'/chat/completions',json=payload,timeout=(15,self.c.get('timeout_s',300)))
        if not response.ok: raise RuntimeError(f'HTTP {response.status_code}: {response.text[:1500]}')
        data=response.json(); choice=data['choices'][0]; raw=choice['message']['content']
        if choice.get('finish_reason')=='length': raise ValueError('Generation hit max_tokens; retry with a larger bounded token budget')
        value,repairs=clean_json(raw)
        return {'value':value,'raw_text':raw,'json_repairs':repairs,'usage':data.get('usage',{}),'elapsed_s':time.perf_counter()-start,'finish_reason':choice.get('finish_reason')}
