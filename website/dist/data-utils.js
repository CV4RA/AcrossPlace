export const MODELS = {
  Z_clip32: ['CLIP ViT-B/32', 'Pretrained'],
  Z_clip16: ['CLIP ViT-B/16', 'Pretrained'],
  Z3_siglip: ['SigLIP2', 'Pretrained'],
  Z4_blip_itc: ['BLIP ITC', 'Pretrained'],
  T1_clip16_place: ['CLIP ViT-B/16 + place', 'Place-supervised · LoRA'],
  T2_lavpr_blip_ms: ['BLIP + MS', 'Adapted LaVPR-style baseline'],
  T3_siglip_ms: ['SigLIP2 + MS', 'Place-supervised · LoRA'],
  T4_siglip_place: ['SigLIP2 + place', 'Place-supervised · LoRA + residual heads'],
};
export const DEMO_MODELS={pretrained_blip:'BLIP · pretrained',pretrained_siglip:'SigLIP2 · pretrained',supervised_siglip:'SigLIP2 · place-supervised',supervised_siglip_P1_s42:'SigLIP2 · place-supervised'};
export function tableRows(metrics, variant, k){
  const groups=new Map();
  for(const r of metrics){
    if(!MODELS[r.method]||r.split!=='test'||r.variant!==variant) continue;
    if(!groups.has(r.method))groups.set(r.method,{id:r.method, name:MODELS[r.method][0],kind:MODELS[r.method][1]});
    const row=groups.get(r.method);
    row[r.direction]={mean:r[`R@${k}_mean`],sd:r[`R@${k}_std`],n:r.n};
  }
  return [...groups.values()].sort((a,b)=>(b.T2I?.mean??-1)-(a.T2I?.mean??-1));
}
export function eligibleCases(cases,direction){return cases.filter(c=>c.direction===direction);}
export function scaleSeries(rows){
  return ['pretrained_blip','pretrained_siglip','supervised_siglip'].map(family=>({family,points:rows.filter(r=>r.protocol==='P1'&&r.direction==='T2I'&&r.k===1&&r.family===family).sort((a,b)=>a.gallery_places-b.gallery_places)}));
}
