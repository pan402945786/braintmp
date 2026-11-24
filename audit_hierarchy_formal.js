'use strict';
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { paired } = require('./synaptic_experiment');
const names = ['pooled','hierarchical','hierarchical-delta','hierarchical-motion','hierarchical-feedback'];
const prefix = process.argv[2];
if (!prefix) throw Error('Supply run directory prefix');
const dirs = names.map(n => path.join('runs', `${prefix}-${n}-20261124`));
const read = (d, p) => JSON.parse(fs.readFileSync(path.join(d,p),'utf8'));
const summaries = dirs.map(d => read(d,'summary.json'));
const seeds = summaries[0].seeds.map(x=>x.seed);
assert.equal(new Set(seeds).size,10);
for (let i=0;i<names.length;i++) {
 const s=summaries[i], c=s.config;
 assert.equal(c.encoder,names[i]); assert.equal(c.profile,'formal'); assert.equal(c.seeds,10); assert.equal(c.samples,2000); assert.equal(c.train,8192);
 assert.deepEqual(s.seeds.map(x=>x.seed),seeds); assert.equal(s.codeHash,summaries[0].codeHash);
}
const reports=dirs.map(d=>seeds.map(seed=>read(d,`seed-${seed}/report.json`)));
for(let j=0;j<10;j++) {
 let reference;
 for(let i=0;i<names.length;i++) {
  const r=reports[i][j], score=r.checkpoints[r.latest].methods.fixed.scores[0];
  assert.equal(r.latest,8192); assert.equal(r.physics[0],1); assert.equal(score.events,2000); assert.equal(score.predictions.length,2000);
  assert.ok(Number.isFinite(score.bits));
  for(const p of score.predictions) {assert.ok(Number.isFinite(p.bits)); assert.ok(p.probabilities.every(Number.isFinite)); assert.ok(Math.abs(p.probabilities.reduce((a,b)=>a+b,0)-1)<1e-5);}
  const pairs=score.predictions.map(x=>[x.target,x.cutoff]);
  if(reference) assert.deepEqual(pairs,reference); else reference=pairs;
 }
}
const results=names.map((name,i)=>({name,meanBits:summaries[i].seeds.reduce((a,x)=>a+x.methods.fixed.bits,0)/10,gate:summaries[i].developmentGate.passed,params:reports[i][0].checkpoints['8192'].methods.fixed.parameterCounts,perSeed:summaries[i].seeds.map(x=>x.methods.fixed.bits)}));
const comparisons=names.slice(1).map((n,i)=>({control:names[i],treatment:n, ...paired(reports[i].map((r,j)=>r.checkpoints['8192'].methods.fixed.scores[0].bits-reports[i+1][j].checkpoints['8192'].methods.fixed.scores[0].bits),summaries[0].config.seed)}));
const result={codeHash:summaries[0].codeHash,seeds,checks:'All 50 reports: matching seed/target/cutoff, 2000 predictions, normalized finite probabilities, physical oracle, training count; runner also validates deployment readback.',results,comparisons};
console.log(JSON.stringify(result,null,2));
