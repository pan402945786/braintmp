'use strict';
// Compare the same method in two separately preregistered, paired studies.
const fs = require('node:fs');
const path = require('node:path');
const { paired } = require('./synaptic_experiment');
const [controlDir, treatmentDir, method = 'fixed'] = process.argv.slice(2);
if (!controlDir || !treatmentDir) throw new Error('Usage: node compare_hierarchies.js CONTROL_DIR TREATMENT_DIR [METHOD]');
const read = (dir, file) => JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
const control = read(controlDir, 'summary.json'), treatment = read(treatmentDir, 'summary.json');
const a = control.config, b = treatment.config;
for (const key of ['profile', 'seed', 'seeds', 'train', 'samples', 'rate', 'memory']) {
  if (a[key] !== b[key]) throw new Error(`Unpaired study configuration: ${key}`);
}
if (control.codeHash !== treatment.codeHash) throw new Error('Source hashes differ; comparisons require identical source');
const seedsA = control.seeds.map(row => row.seed), seedsB = treatment.seeds.map(row => row.seed);
if (JSON.stringify(seedsA) !== JSON.stringify(seedsB)) throw new Error('Seed streams differ');
const rows = seedsA.map(seed => {
  const left = read(controlDir, `seed-${seed}/report.json`), right = read(treatmentDir, `seed-${seed}/report.json`);
  const lastA = left.checkpoints[left.latest].methods[method], lastB = right.checkpoints[right.latest].methods[method];
  if (!lastA || !lastB) throw new Error(`Missing method ${method}`);
  const budget = Math.min(lastA.work, lastB.work);
  const select = report => {
    const points = Object.entries(report.checkpoints).map(([events, point]) => ({ events: Number(events), row: point.methods[method] }));
    return points.filter(point => point.row.work <= budget).sort((x, y) => y.events - x.events)[0];
  };
  const matchedA = select(left), matchedB = select(right);
  return { seed, control: { events: left.latest, bits: lastA.scores[0].bits, work: lastA.work, parameters: lastA.parameterCounts },
    treatment: { events: right.latest, bits: lastB.scores[0].bits, work: lastB.work, parameters: lastB.parameterCounts },
    budget, matchedControl: { events: matchedA.events, bits: matchedA.row.scores[0].bits, work: matchedA.row.work },
    matchedTreatment: { events: matchedB.events, bits: matchedB.row.scores[0].bits, work: matchedB.row.work } };
});
const sameEvents = paired(rows.map(r => r.control.bits - r.treatment.bits), a.seed);
const checkpointBudget = paired(rows.map(r => r.matchedControl.bits - r.matchedTreatment.bits), a.seed);
console.log(JSON.stringify({ control: a.encoder, treatment: b.encoder, method, rows, sameEvents, checkpointBudget,
  note: 'Positive differences favor treatment. Budget comparison uses the last saved checkpoint at or below each seed’s shared budget, so unused budget may differ. Intervals with few seeds are exploratory.' }, null, 2));
