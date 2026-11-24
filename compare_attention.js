'use strict';

// Reuses the source/config pairing checks of compare_hierarchies.js.
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const [motion, feedback, attention, output] = process.argv.slice(2);
if (!motion || !feedback || !attention || !output) {
  throw new Error('Usage: node compare_attention.js MOTION_DIR FEEDBACK_DIR ATTENTION_DIR OUTPUT_JSON');
}
const compare = control => JSON.parse(execFileSync(process.execPath,
  [path.join(__dirname, 'compare_hierarchies.js'), control, attention], { encoding: 'utf8' }));
const comparisons = { vsMotion: compare(motion), vsFeedback: compare(feedback) };
for (const result of Object.values(comparisons)) {
  result.developmentCriterion = {
    meanImprovementThresholdBits: 0.05,
    positiveSeeds: result.sameEvents.perSeed.filter(value => value > 0).length,
    totalSeeds: result.rows.length,
    passed: result.sameEvents.mean >= 0.05 && result.sameEvents.perSeed.every(value => value > 0),
  };
}
const read = (dir, filename) => JSON.parse(fs.readFileSync(path.join(dir, filename), 'utf8'));
const summary = read(attention, 'summary.json');
const reports = Object.fromEntries(Object.entries({ motion, feedback, attention }).map(([name, dir]) => [name,
  summary.seeds.map(({ seed }) => read(dir, `seed-${seed}/report.json`))]));
const checkpoints = Object.keys(reports.attention[0].checkpoints);
for (const group of Object.values(reports)) for (const report of group) {
  if (JSON.stringify(Object.keys(report.checkpoints)) !== JSON.stringify(checkpoints)) throw new Error('Checkpoint schedules differ');
}
const mean = values => values.reduce((sum, x) => sum + x, 0) / values.length;
const learningCurves = checkpoints.map(events => ({ events: Number(events),
  meanBits: Object.fromEntries(Object.entries(reports).map(([name, group]) => [name,
    mean(group.map(report => report.checkpoints[events].methods.fixed.scores[0].bits))])),
  perSeed: summary.seeds.map(({ seed }, i) => ({ seed,
    bits: Object.fromEntries(Object.entries(reports).map(([name, group]) => [name, group[i].checkpoints[events].methods.fixed.scores[0].bits])),
  })),
}));
const diagnostics = summary.seeds.map(({ seed }) => ({ seed,
  attention: read(attention, `seed-${seed}/report.json`).final.fixed.attentionDiagnostics,
  feedback: read(feedback, `seed-${seed}/report.json`).final.fixed.attentionDiagnostics,
}));
const result = { codeHash: summary.codeHash, config: summary.config,
  directories: { motion, feedback, attention }, comparisons, learningCurves, diagnostics,
  note: 'Development only. Primary comparison is vsMotion; positive differences favor attention. Three-seed bootstrap intervals are unstable. Gate ablations are post-training interventions on at most 32 reused development scenes. Event-matched comparisons do not establish equal compute.' };
fs.writeFileSync(output, JSON.stringify(result, null, 2), { flag: 'wx' });
console.log(JSON.stringify({ output, comparisons: Object.fromEntries(Object.entries(comparisons).map(([name, row]) => [name,
  { ...row.sameEvents, criterion: row.developmentCriterion }])) }, null, 2));
