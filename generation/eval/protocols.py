from pathlib import Path
from PIL import Image
import numpy as np

def dhash(path):
    with Image.open(path) as im: a=np.asarray(im.convert('L').resize((9,8)))
    return np.packbits(a[:,1:]>a[:,:-1]).tobytes()

def hamming(a,b): return sum((x^y).bit_count() for x,y in zip(a,b))

class HashTree:
    def __init__(self): self.root=None
    def add(self,value):
        if self.root is None: self.root=(value,{}); return
        node=self.root
        while True:
            d=(value^node[0]).bit_count()
            if not d: return
            if d not in node[1]: node[1][d]=(value,{}); return
            node=node[1][d]
    def near(self,value,radius):
        if self.root is None: return False
        queue=[self.root]
        while queue:
            v,children=queue.pop(); d=(value^v).bit_count()
            if d<=radius: return True
            queue.extend(n for edge,n in children.items() if d-radius<=edge<=d+radius)
        return False

def select_protocol(rows,protocol='B',duplicate_threshold=4):
    if protocol not in ['A','B','C']: raise ValueError('Unknown protocol')
    selected=[]; skipped=[]; hashes={}; tree=HashTree(); global_sources=set(); global_panos=set()
    if protocol!='A':
        for row in rows:
            sources=set(row.get('source_image_paths',[]))|{p for t in row['texts'] for p in t['source_images']}
            global_sources.update(sources)
            global_panos.update(im['panoid'] for im in row['images'] if im['path'] in sources)
        for path in sorted(global_sources):
            hashes[path]=dhash(path); tree.add(int.from_bytes(hashes[path],'big'))
    for row in rows:
        texts=[t for t in row['texts'] if t.get('accepted',True)]
        sources=set(row.get('source_image_paths',[]))|{p for t in texts for p in t['source_images']}
        if not texts or not sources: skipped.append(row['place_id']); continue
        by_path={im['path']:im for im in row['images']}
        if not sources<=set(by_path): raise ValueError('Text evidence is absent from image metadata')
        if protocol=='A': targets=sorted(sources)
        else:
            targets=row.get('target_image_paths',[])
            if set(targets)&sources: raise ValueError('Protocol leakage: source image in target database')
            if not targets: skipped.append(row['place_id']); continue
            source_panos={by_path[p]['panoid'] for p in sources}
            source_dates={(by_path[p].get('year'),by_path[p].get('month')) for p in sources}
            if protocol=='C' and any(None in d for d in source_dates): raise ValueError('C needs reliable capture dates')
            clean=[]
            for path in targets:
                im=by_path[path]
                if path in global_sources or im['panoid'] in global_panos: continue
                if protocol=='C':
                    date=(im.get('year'),im.get('month'))
                    if None in date: raise ValueError('C needs capture dates')
                    if date in source_dates: continue
                for p in [path,*sources]:
                    if p not in hashes: hashes[p]=dhash(p)
                if tree.near(int.from_bytes(hashes[path],'big'),duplicate_threshold): continue
                clean.append(path)
            targets=clean
        if not targets: skipped.append(row['place_id']); continue
        selected.append(dict(row,texts=texts,evaluation_image_paths=targets))
    return selected,skipped
