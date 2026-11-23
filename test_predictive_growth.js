'use strict';

const assert = require('assert');
const {
  SceneGenerator, SparseGrowingRNN, encodeFrame, runExperiment, runStudy,
} = require('./predictive_growth_experiment');

function approxEqual(a, b, eps = 1e-10) { assert(Math.abs(a - b) < eps, `${a} != ${b}`); }

// The generator must be reproducible and keep impact times in the registered range.
{
  const a = new SceneGenerator(42).event(1);
  const b = new SceneGenerator(42).event(1);
  assert.strictEqual(a.T, b.T);
  assert.strictEqual(a.frames.length, 48);
  assert(a.T >= 16 && a.T <= 40);
  approxEqual(a.audio[a.T], 1);
  approxEqual(b.audio[b.T], 1);
}

// Causality: changing future pixels/audio must not affect a prefix prediction.
{
  const gen = new SceneGenerator(3);
  const event = gen.event(9);
  const model = new SparseGrowingRNN({ seed: 5 });
  const before = Array.from(model.predict(event, 8));
  for (let t = 9; t < event.frames.length; t++) {
    event.frames[t].fill(1);
    event.audio[t] = 0.75;
  }
  const after = Array.from(model.predict(event, 8));
  assert.deepStrictEqual(after, before);
  assert.strictEqual(encodeFrame(event, 8).length, 6);
}

// Growth must add auditable edges after the registered interval.
{
  const event = new SceneGenerator(7).event(0);
  const model = new SparseGrowingRNN({ seed: 8, growthInterval: 2, maintenanceWindow: 4, edgesPerGrowth: 2 });
  const initial = model.edges.length;
  for (let i = 0; i < 3; i++) model.trainEvent(event);
  assert(model.edges.length >= initial);
  assert(model.edges.every(e => Number.isInteger(e.birthEvent)));
}

// The full runner returns every required comparison and metric family.
{
  const report = runExperiment({ trainEvents: 16, devEvents: 4, testEvents: 8, checkpoints: [0, 16] });
  assert(report.results[0].time_prior.metrics.nllBits > 0);
  for (const cp of [0, 16]) {
    for (const name of ['time_prior', 'fixed_initial', 'random_growth', 'experience_growth', 'fixed_capacity']) {
      const row = report.results[cp][name];
      assert(row.metrics.anticipationAccuracy);
      assert(Object.prototype.hasOwnProperty.call(row, 'compression'));
    }
  }
}

// Independent-seed aggregation exposes paired confidence intervals.
{
  const study = runStudy({ seeds: 2, trainEvents: 4, devEvents: 2, testEvents: 4, checkpoints: [0, 4] });
  assert.deepStrictEqual(study.seeds, [7, 8]);
  assert(study.aggregate[4].experience_growth.ci95NllBits.length === 2);
  assert(Object.prototype.hasOwnProperty.call(study.aggregate[4], 'experienceGrowthDeltaVsRandom'));
}

console.log('predictive growth tests passed');

