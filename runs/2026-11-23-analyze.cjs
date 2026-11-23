'use strict';

const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { paired } = require('../synaptic_experiment');
const { unpack, loss, LN2, NODES } = require('../synaptic_network');
const { FIRST, CLASSES } = require('../synaptic_scene');

const mean = xs => xs.reduce((sum, x) => sum + x, 0) / xs.length;
const median = xs => {
  const sorted = xs.toSorted((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
};
const json = file => JSON.parse(fs.readFileSync(file, 'utf8'));

function topology(edges) {
  const out = Array.from({ length: NODES }, () => []);
  const indegree = Array(NODES).fill(0);
  for (const edge of edges) {
    out[edge.i].push(edge.j);
    indegree[edge.j]++;
  }
  const reachable = out.map((_, start) => {
    const seen = new Set([start]);
    for (const i of seen) for (const j of out[i]) seen.add(j);
    return seen.size;
  });
  return { stronglyConnected: reachable.every(n => n === NODES), maxIn: Math.max(...indegree), minIn: Math.min(...indegree) };
}

function analyzePredictions(predictions, shared, bias) {
  assert.equal(predictions.length, shared.length);
  const bins = Array.from({ length: 10 }, () => ({ events: 0, confidence: 0, correct: 0 }));
  let biasTotal = 0;
  predictions.forEach(({ bits, target, cutoff, probabilities }, idx) => {
    assert.equal(target, shared[idx].target);
    assert.equal(cutoff, shared[idx].cutoff);
    assert.ok(cutoff >= 4 && cutoff <= 12 && target >= FIRST && target < FIRST + CLASSES);
    assert.equal(probabilities.length, CLASSES);
    assert.ok(probabilities.every(p => Number.isFinite(p) && p > 0 && p <= 1));
    assert.ok(Math.abs(probabilities.reduce((sum, p) => sum + p, 0) - 1) < 1e-10);
    assert.ok(Math.abs(bits + Math.log2(probabilities[target - FIRST])) < 1e-10);
    const confidence = Math.max(...probabilities);
    const bin = bins[Math.min(9, Math.floor(confidence * 10))];
    bin.events++;
    bin.confidence += confidence;
    bin.correct += Number(probabilities.indexOf(confidence) === target - FIRST);
    biasTotal += loss(bias, target - FIRST) / LN2;
  });
  return { ece: bins.reduce((total, b) => total + Math.abs(b.confidence - b.correct), 0) / predictions.length, biasBits: biasTotal / predictions.length };
}

function examineRun(dir) {
  const summary = json(path.join(dir, 'summary.json'));
  const protocol = json(path.join(dir, 'protocol.json'));
  const { config } = protocol;
  const { methods } = config;
  assert.deepEqual(summary.config, config);
  assert.equal(summary.codeHash, protocol.codeHash);
  assert.equal(summary.seeds.length, config.seeds);
  assert.equal(new Set(summary.seeds.map(row => row.seed)).size, config.seeds);
  const trials = summary.seeds.map(({ seed, methods: expected }) => {
    const seedDir = path.join(dir, `seed-${seed}`);
    const report = json(path.join(seedDir, 'report.json'));
    assert.equal(report.seed, seed);
    assert.ok(report.physics.every(rate => rate === 1));
    const point = report.checkpoints[report.latest];
    const shared = point.methods[methods[0]].scores[0].predictions;
    if (config.profile === 'formal') assert.equal(shared.length, config.samples);
    const networks = Object.fromEntries(methods.map(name => {
      const measurements = point.methods[name];
      const score = measurements.scores[0];
      assert.equal(score.bits, expected[name].bits);
      const final = report.final[name];
      const encoded = fs.readFileSync(path.join(seedDir, final.model));
      assert.equal(encoded.length, measurements.deployBytes);
      const model = unpack(encoded);
      assert.equal(model.edges.length, measurements.activeEdges);
      assert.ok(topology(model.edges).stronglyConnected);
      const audit = `${name}-edges.json`;
      const archive = fs.existsSync(path.join(seedDir, audit)) ? json(path.join(seedDir, audit)) : null;
      const newEdges = archive?.edges.filter(edge => edge.born > 0) ?? [];
      if (archive) {
        assert.equal(new Set(archive.edges.map(e => e.id)).size, archive.edges.length);
        assert.equal(archive.edges.filter(e => e.deleted === null).length, model.edges.length);
      }
      const matched = report.compute?.matched[name];
      if (matched) {
        assert.ok(matched.work <= report.compute.budget);
        assert.ok(matched.events <= report.latest);
      }
      return [name, {
        bits: score.bits, absoluteError: score.medianAbsoluteError,
        anticipation: score.anticipation,
        active: measurements.activeEdges, mature: measurements.mature,
        bytes: measurements.deployBytes, trainBytes: final.trainingStateBytes, auditBytes: final.auditBytes,
        topology: topology(model.edges),
        newEdges: { born: newEdges.length, deleted: newEdges.filter(e => e.deleted !== null).length },
        ms: measurements.trainMs + measurements.maintenanceMs, work: measurements.work,
        shuffledBits: final.shuffledBits,
        matchedBits: matched?.scores[0].bits, matchedEvents: matched?.events, matchedWork: matched?.work,
        ...(shared ? analyzePredictions(score.predictions, shared, model.read(new Float64Array(NODES))) : {}),
        compression: final.compressionVsRandom,
        memory: report.memory?.methods[name],
      }];
    }));
    const blocking = report.final.joint?.blocking?.map(({ regime, status, results = [] }) => ({ regime, status, results: results.map(result => {
      assert.equal(result.controls.length, 20);
      assert.ok(result.controls.every(control => control.edges.length === result.edges.length));
      assert.ok(result.restoredDeltaBits.every(delta => delta === 0));
      const control = result.deltaBits.map((_, i) => mean(result.controls.map(row => row.deltaBits[i])));
      return { batch: result.batch, edges: result.edges, target: result.deltaBits, control, difference: result.deltaBits.map((delta, i) => delta - control[i]) };
    }) }));
    return { seed, prior: point.prior[0].bits, uniform: report.uniform.bits, budget: report.compute?.budget, networks, blocking };
  });
  const methodsSummary = Object.fromEntries(methods.map(name => {
    const rows = trials.map(row => row.networks[name]);
    const avg = key => rows[0][key] === undefined ? undefined : mean(rows.map(row => row[key]));
    const memKeys = ['forgettingBits', 'bAfterB', 'aAfterReturn', 'constantA', 'constantB', 'freshA'];
    return [name, { bits: avg('bits'), absoluteError: avg('absoluteError'), anticipation: Object.fromEntries(Object.keys(rows[0].anticipation).map(k => [k, mean(rows.map(row => row.anticipation[k]))])),
      active: avg('active'), mature: avg('mature'), bytes: avg('bytes'), trainBytes: avg('trainBytes'), auditBytes: avg('auditBytes'),
      ms: avg('ms'), work: avg('work'), ece: avg('ece'), shuffledBits: avg('shuffledBits'), biasBits: avg('biasBits'),
      matchedBits: avg('matchedBits'), matchedEvents: avg('matchedEvents'), medianMatchedEvents: rows[0].matchedEvents === undefined ? undefined : median(rows.map(row => row.matchedEvents)),
      memory: rows[0].memory ? { ...Object.fromEntries(memKeys.map(k => [k, mean(rows.map(row => row.memory[k]))])), recoveredByEvents: rows.map(row => row.memory.recoveredByEvents) } : undefined,
    }];
  }));
  const jointVsRandomCompute = config.computeMatch && methods.includes('joint') && methods.includes('random')
    ? paired(trials.map(row => row.networks.random.matchedBits - row.networks.joint.matchedBits), config.seed) : undefined;
  return { config, prior: mean(trials.map(row => row.prior)), uniform: mean(trials.map(row => row.uniform)), methods: methodsSummary,
    jointVsRandomCompute, trials };
}

for (const input of process.argv.slice(2)) {
  const dir = path.resolve(input);
  const analyzed = examineRun(dir);
  const target = path.join(dir, 'analysis.json');
  fs.writeFileSync(target, JSON.stringify(analyzed, null, 2));
  console.log(target);
}
