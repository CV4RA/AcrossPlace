"""Build a checked coarse index and query it locally with text or an image."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageOps
from place_retrieval.data import records, verify_formal
from place_retrieval.encoding import encode_split
from place_retrieval.evaluation import checkpoint_state
from place_retrieval.io import digest, load_config, save_torch, sha256
from place_retrieval.models import context_length, raw_features

def identity(c,checkpoint,manifest):
    return {'config':digest(c),'checkpoint_sha256':sha256(checkpoint),'formal_fingerprint':manifest['fingerprint']}

@torch.inference_mode()
def run(args):
    c=load_config(args.config);torch.set_num_threads(c.get('cpu_threads',8))
    manifest=verify_formal(c['data']);device=torch.device(c['device'])
    expected=identity(c,args.checkpoint,manifest)
    # Validate an existing index before loading model weights.
    bank=None
    if args.command=='query':
        bank=torch.load(args.index,map_location='cpu',weights_only=True)
        if bank.get('identity')!=expected:raise ValueError('Index data/configuration/checkpoint identity differs')
        if args.top_k<1:raise ValueError('top-k must be positive')
    elif Path(args.index).exists():
        raise FileExistsError('Index already exists; choose a new output path')
    model,processor,heads,_=checkpoint_state(Path(args.checkpoint),c,manifest,device)
    model.eval();heads.eval()
    if args.command=='build-index':
        rows=records(c['data'],args.split)
        features=encode_split(model,processor,heads,rows,c,device)
        save_torch(args.index,{'format':'acrossplace-coarse-index-v1','identity':expected,'split':args.split,'features':features,'records':rows})
        print(json.dumps({'index':str(args.index),'images':len(rows['image']),'texts':len(rows['text']),'identity':expected},indent=2));return
    if args.text:
        modality='text';gallery='image'
        inputs=processor(text=[args.text],padding='max_length',max_length=context_length(model,c['training']),truncation=True,return_tensors='pt').to(device)
    else:
        modality='image';gallery='text'
        with Image.open(args.image) as im:inputs=processor(images=ImageOps.exif_transpose(im).convert('RGB'),return_tensors='pt')['pixel_values'].to(device)
    with torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        raw=raw_features(model,inputs,modality).float()
    query=heads.encode(raw,modality).cpu()
    scores=(query@bank['features'][gallery].T)[0]
    # Stable order is the archived record order, including ties.
    indexes=torch.argsort(scores,descending=True,stable=True)[:args.top_k].tolist()
    result=[{'rank':rank+1,'cosine':float(scores[i]),'record':bank['records'][gallery][i]} for rank,i in enumerate(indexes)]
    print(json.dumps({'direction':'T2I' if modality=='text' else 'I2T','stage':'coarse_only','split':bank['split'],'results':result},indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name in ('build-index','query'):
        s=subs.add_parser(name);s.add_argument('--config',required=True);s.add_argument('--checkpoint',required=True);s.add_argument('--index',required=True)
        if name=='build-index':s.add_argument('--split',choices=('train','val','test'),default='test')
        else:
            q=s.add_mutually_exclusive_group(required=True);q.add_argument('--text');q.add_argument('--image');s.add_argument('--top-k',type=int,default=5)
    run(p.parse_args())
