import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {tableRows,eligibleCases,scaleSeries} from '../dist/data-utils.js';
const d=JSON.parse(readFileSync(new URL('../dist/data.json',import.meta.url)));
test('public leaderboard agrees with archived test evidence, not illustrative scores',()=>{
 const rows=tableRows(d.metrics.P0,'coarse_only',1);assert.equal(rows[0].id,'T2_lavpr_blip_ms');assert.ok(Math.abs(rows[0].T2I.mean-17.544)<.001);assert.ok(Math.abs(rows[0].I2T.mean-23.617)<.001);
 for(const protocol of ['P0','P1'])for(const variant of ['coarse_only','selected_model'])for(const k of [1,5,10])for(const r of tableRows(d.metrics[protocol],variant,k))for(const dir of ['T2I','I2T'])if(r[dir])assert.ok(r[dir].mean>=0&&r[dir].mean<=100);
});
test('case directions, real assets and place relevance are consistent',()=>{
 for(const dir of ['T2I','I2T'])assert.equal(eligibleCases(d.cases,dir).length,6);
 for(const c of d.cases){for(const [m,p]of Object.entries(c.predictions)){assert.equal(p.hit,p.place===c.place);if(c.direction==='T2I')assert.ok(existsSync(new URL('../dist/'+p.image,import.meta.url)));else assert.ok(p.text.length>0);}}
 for(const p of d.examples){assert.ok(p.source.length&&p.target.length);assert.equal(p.source.filter(x=>p.target.includes(x)).length,0);for(const url of [...p.source,...p.target])assert.ok(existsSync(new URL('../dist/'+url,import.meta.url)));}
});
test('scale chart uses fixed-query P1 diagnostic and includes full gallery',()=>{for(const s of scaleSeries(d.scale)){assert.equal(s.points[0].gallery_places,500);assert.equal(s.points.at(-1).gallery_places,8425);}});
