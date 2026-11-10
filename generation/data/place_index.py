import random
import numpy as np
from scipy.spatial import cKDTree

def haversine(a,b):
    a,b=np.radians(a),np.radians(b)
    d=b-a
    return 6371000*2*np.arcsin(np.sqrt(np.clip(np.sin(d[...,0]/2)**2+np.cos(a[...,0])*np.cos(b[...,0])*np.sin(d[...,1]/2)**2,0,1)))

def split_places(places, seed=42, ratios=(.8,.1,.1), buffer_m=25, mode='place'):
    if len(ratios)!=3 or min(ratios)<0 or not np.isclose(sum(ratios),1): raise ValueError('Ratios must sum to one')
    parent=list(range(len(places)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b): parent[find(a)]=find(b)
    # Every observed GPS participates, not just the centroid. Connected neighbors
    # remain in the same split, including aliases of the same physical location.
    points=[]; owners=[]
    for i,p in enumerate(places):
        for gps in p['gps']: points.append(gps); owners.append(i)
    if points and buffer_m>0:
        ll=np.radians(points); xyz=np.stack([np.cos(ll[:,0])*np.cos(ll[:,1]),np.cos(ll[:,0])*np.sin(ll[:,1]),np.sin(ll[:,0])],1)
        for a,b in cKDTree(xyz).query_pairs(2*np.sin(buffer_m/6371000/2),output_type="ndarray"):
            union(owners[a],owners[b])
    if mode=='region':
        # Cities are explicit geographic regions; never mix a city between splits.
        city={}
        for i,p in enumerate(places):
            c=p['metadata']['city']
            if c in city: union(i,city[c])
            city[c]=i
    elif mode!='place': raise ValueError('mode must be place or region')
    groups={}
    for i in range(len(places)): groups.setdefault(find(i),[]).append(i)
    groups=list(groups.values()); random.Random(seed).shuffle(groups)
    counts=[0,0,0]; targets=np.array(ratios)*len(places)
    for group in groups:
        s=int(np.argmax(targets-np.array(counts)))
        for i in group: places[i]['split']=['train','val','test'][s]
        counts[s]+=len(group)
    return places, {'seed':seed,'ratios':list(ratios),'buffer_m':buffer_m,'mode':mode,'components':len(groups),'counts':dict(zip(['train','val','test'],counts)), 'spatial_check':'all cross-split observed GPS pairs are farther than buffer_m'}
