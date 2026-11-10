"""Qwen3-VL official HF interfaces; imported lazily for CPU-only data tooling."""
import json,importlib.metadata
from pathlib import Path
import torch
from PIL import Image,ImageOps
from utils.gpu_monitor import GPUMonitor

def clean_json(text):
    candidate=text.strip(); start=candidate.find('{')
    if start<0: raise ValueError('Response contains no JSON object')
    candidate=candidate[start:]; repairs=[]
    def decode(value):
        obj,end=json.JSONDecoder().raw_decode(value)
        if not isinstance(obj,dict): raise ValueError('Expected JSON object')
        return obj
    try: obj=decode(candidate)
    except json.JSONDecodeError:
        # Remove trailing commas OUTSIDE strings only; never invent missing facts.
        out=[]; quoted=False; escaped=False
        for i,ch in enumerate(candidate):
            if quoted:
                out.append(ch)
                if escaped: escaped=False
                elif ch=='\\': escaped=True
                elif ch=='"': quoted=False
            else:
                if ch=='"': quoted=True
                if ch==',':
                    tail=candidate[i+1:].lstrip()
                    if tail.startswith(('}',']')): continue
                out.append(ch)
        repaired=''.join(out)
        obj=decode(repaired); repairs.append('removed_trailing_commas')
    if text.strip()!=candidate: repairs.append('removed_wrapper')
    return obj,repairs

def resolve_model_path(c):
    name=c.get('name','Qwen/Qwen3-VL-4B-Instruct')
    if Path(name).is_dir(): return str(Path(name).resolve())
    if Path(name).is_absolute() or name.startswith(('.', '~')) or ':' in name:
        raise FileNotFoundError(f'Local model directory does not exist: {name}. Set model.name to the downloaded Qwen3-VL snapshot.')
    from huggingface_hub import snapshot_download,try_to_load_from_cache
    cache_dir=c.get('cache_dir','outputs/models_cache'); revision=c.get('revision','main')
    if c.get('local_files_only',True):
        # Look up existing snapshots read-only; no AutoProcessor network probes.
        for cache in [cache_dir,None]:
            config=try_to_load_from_cache(name,'config.json',revision=revision,cache_dir=cache)
            if isinstance(config,str): return str(Path(config).resolve().parent)
    try: return snapshot_download(name,revision=revision,cache_dir=cache_dir,local_files_only=c.get('local_files_only',True))
    except Exception as e:
        raise FileNotFoundError(f'No usable local snapshot for {name}. Download Qwen3-VL weights to a local directory and set model.name to that path; default mode never downloads model weights. Details: {e}') from e

def load_components(c):
    version=importlib.metadata.version('transformers')
    try: from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,BitsAndBytesConfig
    except ImportError as e: raise RuntimeError(f'Transformers {version} lacks an importable Qwen3VL interface. Use .venv-qwen3 and requirements.txt; see original error: {e}') from e
    device=c.get('device','cuda')
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable. Set model.device: cpu for explicit CPU inference, or run --dry-run for data checks')
    mode=c.get('dtype','auto'); cuda=device!='cpu' and torch.cuda.is_available()
    dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if cuda else torch.float32
    if mode!='auto': dtype={'float16':torch.float16,'bfloat16':torch.bfloat16,'float32':torch.float32}[mode]
    if not cuda: dtype=torch.float32
    if dtype==torch.bfloat16 and cuda and not torch.cuda.is_bf16_supported(): raise ValueError('GPU does not support bfloat16; choose float16')
    common={'revision':c.get('revision','main'),'local_files_only':c.get('local_files_only',True),'cache_dir':c.get('cache_dir','outputs/models_cache')}
    name=c.get('name',c.get('model','Qwen/Qwen3-VL-4B-Instruct'))
    resolved=resolve_model_path(c)
    # Resolve first to ensure truly offline AutoProcessor loading in 4.57.x.
    processor=AutoProcessor.from_pretrained(resolved,**common)
    processor.tokenizer.padding_side='left'
    # Qwen3-VL image processor uses size shortest_edge/longest_edge as pixel bounds.
    if hasattr(processor,'image_processor'):
        ip=processor.image_processor
        if hasattr(ip,'size'): ip.size={'shortest_edge':c.get('min_image_pixels',4096),'longest_edge':c.get('max_image_pixels',602112)}
    kwargs={'dtype':dtype,'attn_implementation':c.get('attn_implementation','sdpa'),**common}
    device_map='cpu' if not cuda else c.get('device_map','auto'); kwargs['device_map']=device_map
    if cuda and device_map=='auto':
        total=torch.cuda.get_device_properties(0).total_memory
        kwargs['max_memory']={0:c.get('gpu_memory',int(total*.83)),'cpu':c.get('cpu_memory','24GiB')}
    if c.get('load_in_4bit',False):
        if not cuda: raise ValueError('4-bit bitsandbytes mode requires CUDA in this pipeline')
        try: import bitsandbytes
        except ImportError as e: raise RuntimeError('Install requirements-4bit.txt for optional 4-bit inference') from e
        kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    model=Qwen3VLForConditionalGeneration.from_pretrained(resolved,**kwargs).eval()
    return model,processor,{'model':name,'revision':getattr(model.config,'_commit_hash',None) or (Path(resolved).name if Path(resolved).parent.name=='snapshots' else common['revision']),'transformers':version,'dtype':str(dtype),'device_map':getattr(model,'hf_device_map',str(device_map))}

class VLM:
    def __init__(self,c,generation=None):
        self.c=c; self.generation=generation or c; self.monitor=GPUMonitor(); self.last_responses=[]
        with self.monitor.measure('model_load'): self.model,self.processor,self.info=load_components(c)
        self.revision=self.info['revision']
    @torch.inference_mode()
    def generate_batch(self,prompts,images):
        if len(prompts)!=len(images): raise ValueError('Prompts/images length mismatch')
        messages=[]; resized=[]
        for prompt,path in zip(prompts,images):
            content=[]
            if path:
                with Image.open(path) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB'); pixels=self.c.get('max_image_pixels',602112)
                    if im.width*im.height>pixels:
                        ratio=(pixels/(im.width*im.height))**.5; im=im.resize((max(1,int(im.width*ratio)),max(1,int(im.height*ratio))))
                    resized.append(im); content.append({'type':'image','image':im})
            content.append({'type':'text','text':prompt}); messages.append([{'role':'user','content':content}])
        with self.monitor.measure('generation') as metric:
            inputs=self.processor.apply_chat_template(messages,tokenize=True,padding=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
            inputs.pop('token_type_ids',None); inputs=inputs.to(self.model.device)
            sample=self.generation.get('do_sample',False)
            kwargs={'max_new_tokens':self.generation.get('max_new_tokens',512),'do_sample':sample,'use_cache':True}
            if sample: kwargs['temperature']=self.generation.get('temperature',.2)
            else: kwargs.update(temperature=1.0,top_p=1.0,top_k=50)
            out=self.model.generate(**inputs,**kwargs)
            decoded=self.processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        self.last_responses=[{'raw_text':text,'metric':dict(metric)} for text in decoded]
        parsed=[]
        try:
            for text,record in zip(decoded,self.last_responses):
                obj,repairs=clean_json(text); record['json_repairs']=repairs; parsed.append(obj)
            return parsed
        finally:
            del inputs,out,resized
            if self.c.get('clear_cache',True): self.monitor.clear()
    def generate(self,prompt,image=None): return self.generate_batch([prompt],[image])[0]
