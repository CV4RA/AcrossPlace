import {tableRows,eligibleCases,scaleSeries,DEMO_MODELS} from './data-utils.js';
const $=s=>document.querySelector(s);
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
let data,place=0,view='source',protocol='P0',direction='T2I';
function options(node,items){node.replaceChildren(...items.map(([value,label])=>{const o=el('option',null,label);o.value=value;return o;}));}
function image(url,alt){const i=el('img');i.src=url;i.alt=alt;i.loading='lazy';return i;}
function tabs(container,key,value){for(const b of container.querySelectorAll('button')){const on=b.dataset[key]===value;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));}}
function renderPlace(resetText=true){
 const p=data.examples[place];$('#place-id').textContent=p.id;$('#place-counter').textContent=`${String(place+1).padStart(2,'0')} / ${String(data.examples.length).padStart(2,'0')} CURATED PLACES`;
 $('#place-images').replaceChildren(...p[view].map((url,i)=>{const f=el('figure');f.append(image(url,`${p.city}, ${view} view ${i+1}`),el('figcaption',null,`${view.toUpperCase()} VIEW ${String(i+1).padStart(2,'0')} / ${p.id}`));return f;}));
 if(resetText)options($('#text-select'),p.texts.map((t,i)=>[i,`${t.type} · ${t.text_id.split('_').pop()}`]));
 renderText();
}
function renderText(){const t=data.examples[place].texts[Number($('#text-select').value)||0];$('#place-text').textContent=t.text;$('#text-id').textContent=t.text_id;}
function renderTable(){
 const k=Number($('#k-select').value),variant=$('#stage-select').value;
 const rows=tableRows(data.metrics[protocol],variant,k);
 const best=Object.fromEntries(['T2I','I2T'].map(d=>[d,Math.max(...rows.map(r=>r[d]?.mean??-1))]));
 $('#table-caption').textContent=`${protocol} / FIXED TEST SET / RECALL@${k} (%) / ${variant==='coarse_only'?'COARSE RETRIEVAL':'VALIDATION-SELECTED RERANKING'}`;
 $('#leaderboard').replaceChildren(...rows.map(r=>{const tr=el('tr');const name=el('td');name.append(el('span',null,r.name),el('span','model-kind',r.kind));tr.append(name);
 for(const d of ['T2I','I2T']){const td=el('td');if(r[d]){const s=r[d];td.append(el('span',`score${s.mean===best[d]?' best':''}`,s.mean.toFixed(2)),el('span','sd',s.n>1?`± ${s.sd.toFixed(2)}`:''));}else td.textContent='—';tr.append(td);}tr.append(el('td',null,String(r.T2I?.n??r.I2T?.n??'—')));return tr;}));
}
function drawScale(){
 const svg=$('#scale-chart'),ns='http://www.w3.org/2000/svg';const series=scaleSeries(data.scale);const colors=['#d28b58','#82bfc3','#d4ef81'];
 const make=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;};
 const x=n=>62+(n-500)/(8425-500)*550,y=n=>240-n/20*206;
 for(const tick of [0,5,10,15,20]){make('line',{x1:62,x2:612,y1:y(tick),y2:y(tick),stroke:'#30454a','stroke-dasharray':'3 5'});make('text',{x:48,y:y(tick)+4,fill:'#9bb3ad','text-anchor':'end','font-size':10},String(tick));}
 make('text',{x:62,y:17,fill:'#9bb3ad','font-size':10},'T2I R@1 (%)');
 for(const n of [500,2000,4000,8425]){make('text',{x:x(n),y:264,fill:'#9bb3ad','text-anchor':'middle','font-size':10},n.toLocaleString('en'));}
 make('text',{x:612,y:287,fill:'#9bb3ad','text-anchor':'end','font-size':10},'Gallery places');
 $('#chart-legend').replaceChildren(...series.map((s,i)=>{make('polyline',{points:s.points.map(p=>`${x(p.gallery_places)},${y(p.mean)}`).join(' '),fill:'none',stroke:colors[i],'stroke-width':2.5});for(const p of s.points){const c=make('circle',{cx:x(p.gallery_places),cy:y(p.mean),r:4,fill:colors[i]});const title=document.createElementNS(ns,'title');title.textContent=`${s.family}, ${p.gallery_places} places: ${p.mean.toFixed(3)}%`;c.append(title);}
 const label=el('span');const swatch=el('i');swatch.style.setProperty('--c',colors[i]);label.append(swatch,document.createTextNode(['BLIP','SigLIP2','SigLIP2 + place'][i]));return label;}));
 const table=el('table'),head=el('tr');for(const t of ['Model','Gallery places','T2I R@1 (%)'])head.append(el('th',null,t));const thead=el('thead');thead.append(head);table.append(thead);const body=el('tbody');for(const s of series)for(const p of s.points){const r=el('tr');for(const t of [s.family,p.gallery_places,p.mean.toFixed(3)])r.append(el('td',null,String(t)));body.append(r);}table.append(body);$('#chart-values').append(table);
}
function initCases(){const cases=eligibleCases(data.cases,direction);options($('#case-select'),cases.map((c,i)=>[i,`${c.id} · ${c.stratum}`]));renderCase();}
function renderCase(){
 const c=eligibleCases(data.cases,direction)[Number($('#case-select').value)||0];const q=$('#query-panel');q.replaceChildren();
 if(direction==='I2T')q.append(image(c.target,`Query target image of ${c.place}`));
 const copy=el('div','query-copy');copy.append(el('span','mono',direction==='T2I'?'TEXT QUERY / EXACT ANNOTATION':'IMAGE QUERY / INDEPENDENT TARGET'));
 copy.append(el('p',null,direction==='T2I'?`“${c.text}”`:`Which description identifies this place?`),el('small',null,`GROUND-TRUTH PLACE: ${c.place} · ${c.id}`));q.append(copy);
 $('#prediction-grid').replaceChildren(...Object.entries(c.predictions).map(([id,p])=>{const card=el('article','prediction');const head=el('div','prediction-head');head.append(el('h3',null,DEMO_MODELS[id]||id),el('span',`status ${p.hit?'hit':'miss'}`,p.hit?'✓ PLACE MATCH':'× DIFFERENT PLACE'));card.append(head);
 if(direction==='T2I')card.append(image(p.image,`Top-one ${DEMO_MODELS[id]||id} prediction: ${p.place}`));else card.append(el('blockquote',null,`“${p.text}”`));card.append(el('div','prediction-bottom',`TOP 1 / ${p.place}`));return card;}));
}
async function init(){try{const r=await fetch('data.json');if(!r.ok)throw Error(`HTTP ${r.status}`);data=await r.json();
 const h=data.examples.find(p=>p.id==='Rome:0001741')||data.examples[0];$('#hero-source').src=h.source[0];$('#hero-target').src=h.target[0];$('#hero-text').textContent=h.texts.find(t=>t.text_id.endsWith('_T2'))?.text||h.texts[0].text;
 options($('#place-select'),data.examples.map((p,i)=>[i,`${p.city} · ${p.id.split(':')[1]}`]));
 $('#place-select').addEventListener('change',e=>{place=Number(e.target.value);renderPlace();});$('#text-select').addEventListener('change',renderText);
 $('#view-tabs').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;view=b.dataset.view;tabs($('#view-tabs'),'view',view);renderPlace(false);});
 $('#protocol-tabs').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;protocol=b.dataset.protocol;tabs($('#protocol-tabs'),'protocol',protocol);renderTable();});
 $('#stage-select').addEventListener('change',renderTable);$('#k-select').addEventListener('change',renderTable);
 $('#direction-tabs').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;direction=b.dataset.direction;tabs($('#direction-tabs'),'direction',direction);initCases();});$('#case-select').addEventListener('change',renderCase);
 renderPlace();renderTable();drawScale();initCases();
 }catch(e){$('#load-error').hidden=false;console.error('AcrossPlace records unavailable',e);}}
init();
