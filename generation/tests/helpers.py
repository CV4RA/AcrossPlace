"""Synthetic-only construction fixtures; no pretrained weights."""
from pathlib import Path
import numpy as np
from PIL import Image
from utils.io import write_jsonl

def synthetic_annotations(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(42); rows=[]
    for i in range(6):
        images=[]
        for j in range(2):
            path=str((root/f'p{i}_{j}.png').resolve()); Image.fromarray(rng.integers(0,255,(16,16,3),dtype=np.uint8)).save(path)
            images.append({'path':path,'gps':[10+i*.01,20.],'year':2020+j,'month':1,'panoid':f'p{i}v{j}','northdeg':0})
        pid=f'p{i}'; texts=[{'text_id':f'{pid}t{k}','type':kind,'text':f'fixture {i} description {k}','source_images':[images[0]['path']],'accepted':True} for k,kind in enumerate(['detailed','concise','partial','relational'])]
        rows.append({'place_id':pid,'dataset':'SYNTHETIC','split':'train' if i<4 else 'val','images':images,'image_paths':[im['path'] for im in images],'gps':[im['gps'] for im in images],'metadata':{'city':'fixture'},'source_image_paths':[images[0]['path']],'target_image_paths':[images[1]['path']],'texts':texts})
    path=root/'annotations.jsonl'; write_jsonl(path,rows); return str(path)
