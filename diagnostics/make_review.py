"""Render the metadata-selected blinded review packet, without model predictions."""
import argparse
import json
from pathlib import Path
import zipfile
from PIL import Image
from benchmark import read, write, digest, sha


def main(root):
    root = Path(root)
    plan = read(root/'plan.json')
    selected = read(root/'quality_review_private.json')['rows']
    destination = root/'human_review'
    destination.mkdir(exist_ok=True)
    (destination/'images').mkdir(exist_ok=True)
    public = []
    image_manifest = []
    for row in selected:
        item = {k:v for k,v in row.items() if k != 'rule_status'}
        for key in ['source_images','target_images']:
            rels = []
            for canonical in row[key]:
                prefix = plan['canonical_prefix'].rstrip('/')+'/'
                if not canonical.startswith(prefix):
                    raise ValueError('Image outside canonical prefix')
                source = Path(plan['image_root'])/canonical[len(prefix):]
                name = 'images/'+digest(canonical)[:24]+'.jpg'
                output = destination/name
                if not output.exists():
                    with Image.open(source) as image:
                        image = image.convert('RGB')
                        image.thumbnail((960,960))
                        image.save(output, quality=92)
                    image_manifest.append({'canonical':canonical,'source_sha256':sha(source),
                                           'review_image':name,'review_sha256':sha(output)})
                rels.append(name)
            item[key+'_display'] = rels
        public.append(item)
    payload = json.dumps(public, ensure_ascii=False).replace('</','<\\/')
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>AcrossViews 人工盲评</title>
<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 20px;background:#f5f6f8;color:#192333}header{position:sticky;top:0;background:#fff;padding:15px;border-bottom:2px solid #ccd3dd;z-index:2}button,select,input{font:inherit;padding:7px;margin:5px}article{background:white;padding:24px;margin:24px 0;border:1px solid #ccd3dd;border-radius:10px}.views{display:flex;gap:16px;flex-wrap:wrap}.views section{flex:1;min-width:320px}.views img{max-width:100%;max-height:370px}blockquote{font-size:19px;background:#eef3fc;margin:15px 0;padding:15px}.row{display:flex;flex-wrap:wrap;gap:15px}label{display:block}textarea{width:100%;min-height:45px;font:inherit}.hint{font-size:14px;color:#435469}</style>
<header><b>AcrossViews：文本可靠性与跨视图可见性盲评</b><p class="hint">分别评价文本对来源图像是否准确、目标图像是否提供对应证据，以及描述是否过于通用。不要将“来源正确”自动等同“目标可见”。来源/目标中的任意一张图提供证据即可；证据不足请选择无法判断。此页面隐藏自动质量标签和模型检索结果。进度自动保存在当前浏览器；完成后导出 JSON。</p><button id="export">导出审核 JSON</button><span id="progress"></span></header><main id="main"></main>
<script>const rows=PAYLOAD;const storageKey='acrossviews-quality-review-v1';let saved={};try{saved=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){};
const fields={source_correctness:[['supported','完整支持'],['partial','部分正确/有遗漏'],['contradicted','有明确错误'],['unknown','无法判断']],target_visibility:[['visible','目标中证据充分'],['partial','目标中部分可见'],['not_visible','目标中未见对应证据'],['unknown','无法判断']],place_specificity:[['distinctive','具有较具体的地点线索'],['generic','很通用，可能适用多个地点'],['unknown','无法判断']]};
const labels={source_correctness:'① 原文与来源图像是否一致？',target_visibility:'② 原文线索在独立目标图像中是否可见？',place_specificity:'③ 原文是否具有地点辨识性？'};
function update(){localStorage.setItem(storageKey,JSON.stringify(saved));let n=rows.filter(r=>Object.keys(fields).every(k=>saved[r.text_id]?.[k])).length;document.getElementById('progress').textContent=`已完成 ${n}/${rows.length}`;}
for(const r of rows){const a=document.createElement('article');const h=document.createElement('h3');h.textContent=r.audit_id+' · '+r.place_id;a.append(h);const b=document.createElement('blockquote');b.textContent=r.text;a.append(b);const views=document.createElement('div');views.className='views';for(const [key,title] of [['source_images_display','来源图像（用于生成文本）'],['target_images_display','独立目标图像（用于检索）']]){let s=document.createElement('section');let h=document.createElement('h4');h.textContent=title;s.append(h);for(const src of r[key]){let link=document.createElement('a');link.href=src;link.target='_blank';let img=document.createElement('img');img.src=src;img.loading='lazy';link.append(img);s.append(link)}views.append(s)}a.append(views);const controls=document.createElement('div');controls.className='row';for(const [key,options]of Object.entries(fields)){let label=document.createElement('label');label.textContent=labels[key];let sel=document.createElement('select');sel.append(new Option('请选择',''));for(const [value,text]of options)sel.append(new Option(text,value));sel.value=saved[r.text_id]?.[key]||'';sel.onchange=()=>{saved[r.text_id]={...(saved[r.text_id]||{}),[key]:sel.value};update()};label.append(sel);controls.append(label)}a.append(controls);let note=document.createElement('textarea');note.placeholder='备注：指出具体错误、不可见的线索或歧义';note.value=saved[r.text_id]?.comment||'';note.oninput=()=>{saved[r.text_id]={...(saved[r.text_id]||{}),comment:note.value};update()};a.append(note);document.getElementById('main').append(a)}
document.getElementById('export').onclick=()=>{const result={schema_version:1,review_scope:'source_correctness_target_visibility_place_specificity',reviewer_type:'human',exported_at:new Date().toISOString(),rows:rows.map(r=>({...r,...saved[r.text_id]}))};let blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'});let link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='AcrossViews-240-human-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000)};update();</script></html>'''.replace('PAYLOAD',payload)
    (destination/'index.html').write_text(html,encoding='utf-8')
    write(destination/'image_provenance.json',image_manifest)
    write(root/'human_review_sampling.json',{'selected_without_model_scores':True,'protocol':'P1','split':'test',
              'n':len(public),'rule_strata':{'valid':120,'uncertain':120},
              'limitation':'Stratified diagnostic audit; not an unweighted estimate of dataset-wide quality',
              'text_ids':[r['text_id'] for r in public]})
    archive = root/'AcrossViews-240-human-review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(destination.rglob('*')):
            if p.is_file():
                z.write(p,'AcrossViews-240-human-review/'+p.relative_to(destination).as_posix())
    write(archive.with_suffix('.sha256.json'),{'file':archive.name,'sha256':sha(archive),'bytes':archive.stat().st_size})
    print(json.dumps({'review_packet':str(archive),'bytes':archive.stat().st_size,'rows':len(public)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    main(parser.parse_args().root)
