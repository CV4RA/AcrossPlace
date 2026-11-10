"""CSV-authoritative GSV-Cities reader; original images are never modified.
Format: https://github.com/amaralibey/gsv-cities
"""
import csv,random
from pathlib import Path
from PIL import Image
from utils.io import digest
REQUIRED={'place_id','year','month','northdeg','city_id','lat','lon','panoid'}

def parse_gsv(root,cities=None,limit_places=None,images_per_place=None,place_ids=None,seed=42,shuffle=False,skip_bad_images=False,verify_images=False,errors=None):
    root=Path(root); frames=root/'Dataframes'; errors=errors if errors is not None else []
    if not frames.is_dir(): raise FileNotFoundError(f'Missing official Dataframes directory: {frames}')
    if limit_places is not None and limit_places<1: raise ValueError('limit_places must be positive')
    wanted=set(place_ids) if place_ids is not None else None
    files=[frames/(c+'.csv') for c in cities] if cities else sorted(frames.glob('*.csv')); places=[]
    rng=random.Random(seed)
    for file in files:
        with file.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if not REQUIRED<=set(reader.fieldnames or []): raise ValueError(f'Invalid official CSV schema: {file}')
            groups={}
            for row in reader:
                if row['city_id']!=file.stem: raise ValueError(f'City mismatch: {file}')
                groups.setdefault(row['place_id'],[]).append(row)
        groups=sorted(groups.items(),key=lambda x:int(x[0]))
        if shuffle: rng.shuffle(groups)
        directory=root/'Images'/file.stem; names=None
        for local_id,entries in groups:
            pid=f'{file.stem}:{int(local_id):07d}'
            if wanted is not None and pid not in wanted: continue
            if limit_places is not None and len(places)>=limit_places: break
            images=[]
            for row in entries:
                if images_per_place is not None and len(images)>=images_per_place: break
                path=None
                try:
                    stem=f"{file.stem}_{int(local_id):07d}_{int(row['year']):04d}_{int(row['month']):02d}_{int(row['northdeg']):03d}"
                    path=directory/(stem+f"_{row['lat']}_{row['lon']}_{row['panoid']}.jpg")
                    if not path.is_file():
                        if names is None:
                            names={}
                            for candidate in directory.glob('*'):
                                if candidate.suffix.lower() in {'.jpg','.jpeg','.png'}: names.setdefault('_'.join(candidate.name.split('_')[:5]),[]).append(candidate)
                        matches=[p for p in names.get(stem,[]) if p.stem.endswith('_'+row['panoid'])]
                        if len(matches)!=1: raise FileNotFoundError(f'Official row has {len(matches)} matching files: {path}')
                        path=matches[0]
                    lat,lon=float(row['lat']),float(row['lon'])
                    if not (-90<=lat<=90 and -180<=lon<=180): raise ValueError('Invalid GPS')
                    if verify_images:
                        with Image.open(path) as im: im.verify()
                    path=str(path.resolve()); images.append({'image_id':'I_'+digest(path)[:24],'path':path,'gps':[lat,lon],'year':int(row['year']),'month':int(row['month']),'northdeg':int(row['northdeg']),'panoid':row['panoid']})
                except (OSError,ValueError) as e:
                    errors.append({'place_id':pid,'path':str(path),'reason':str(e),'status':'skipped'})
                    if not skip_bad_images: raise
            if images: places.append({'place_id':pid,'dataset':'GSV-Cities','images':images,'image_paths':[im['path'] for im in images],'gps':[im['gps'] for im in images],'metadata':{'city':file.stem,'official_place_id':local_id}})
        if limit_places is not None and len(places)>=limit_places: break
    if not places: raise ValueError('No usable places found; inspect image_errors.jsonl')
    return places
