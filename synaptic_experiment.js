'use strict';

const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');
const { parseArgs } = require('node:util');
const { Worker, isMainThread, workerData } = require('node:worker_threads');
const { FIRST, CLASSES, stream, shuffle, makeEvent, sceneAt, physicalOracle } = require('./synaptic_scene');
const { LN2, GROWTH, SynapticNetwork, unpack, loss } = require('./synaptic_network');

const METHODS = ['fixed', 'random', 'hebb', 'bidirectional', 'joint', 'shuffled', 'capacity'];
const HORIZONS = [2, 4, 6, 8, 10, 12];
const mean = items => items.reduce((total, value) => total + value, 0) / items.length;
const defaults = {
  profile: 'pilot', seed: 7, seeds: 3, train: 1024, samples: 256, checkpoints: [0, 128, 512, 1024],
  rate: 0.003, encoder: 'motion', learner: 'bptt', traceDecay: 1, interval: 64,
  candidates: 128, edgesPerGrowth: 4, maxEdges: 512, maxIn: 20, fixedCapacity: 176,
  confirmEvents: 32, confirmWindows: 3, support: 0.0001, matureRate: 0.3,
  homeostasis: true, prune: true, methods: METHODS, computeMatch: false, memory: false,
};

function modelMedian(p) {
  let sum = 0;
  for (let i = 0; i < p.length; i++) {
    sum += p[i];
    if (sum >= 0.5) return i;
  }
  return p.length - 1;
}

function summarize(rows) {
  const errors = rows.map(row => row.error).sort((a, b) => a - b);
  const anticipation = Object.fromEntries(HORIZONS.map((t, k) => [t, mean(rows.map(row => row.ahead[k]))]));
  let maxAhead = 0;
  for (const t of HORIZONS) {
    if (anticipation[t] < 0.8) break;
    maxAhead = t;
  }
  return { bits: mean(rows.map(row => row.bits)),
    medianAbsoluteError: (errors[Math.floor((errors.length - 1) / 2)] + errors[Math.floor(errors.length / 2)]) / 2,
    anticipation, maxAhead, events: rows.length };
}

function evaluate(model, data, { prior, blocked, pairing, details = false, projections } = {}) {
  const rows = data.map((event, index) => {
    const source = pairing ? data[pairing[index]] : event;
    const stop = pairing ? event.cutoff : event.target + FIRST - HORIZONS[0];
    const states = model?.forward(source.features, stop, blocked, false, projections?.[index]).a;
    const p = prior ?? model.read(states[event.cutoff + 1]);
    return { bits: loss(p, event.target) / LN2, error: Math.abs(modelMedian(p) - event.target),
      ahead: pairing ? HORIZONS.map(() => 0) : HORIZONS.map(t => {
        const q = prior ?? model.read(states[event.target + FIRST - t + 1]);
        return Number(Math.abs(modelMedian(q) - event.target) <= 1);
      }), ...(details ? { target: event.target + FIRST, cutoff: event.cutoff, probabilities: Array.from(p) } : {}) };
  });
  return { ...summarize(rows), ...(details ? { predictions: rows } : {}) };
}

function frequencies(counts) {
  const total = counts.reduce((a, b) => a + b, 0);
  return Float64Array.from(counts, c => c / total);
}

function sections(config) {
  if (!config.memory) return [{ name: 'A', regime: 0, events: config.train }];
  return ['A1', 'B', 'A2'].map((name, k) => ({ name, regime: k === 1 ? 1 : 0, events: config.train }));
}

function evaluationData(config, seed) {
  const profileSplit = { pilot: 20, development: 21, formal: 93 };
  const split = profileSplit[config.profile];
  return Array.from({ length: config.memory ? 2 : 1 }, (_, regime) =>
    Array.from({ length: config.samples }, (_, i) => makeEvent(seed, split, i, config.encoder, regime, config.memory)));
}

function pairPermutation(size, seed) {
  const indices = shuffle(Array.from({ length: size }, (_, i) => i), stream(seed, 410));
  const pairing = [];
  indices.forEach((index, i) => { pairing[index] = indices[(i + 1) % size]; });
  return pairing;
}

function scoreSnapshot(snapshot, data, details = false) {
  const model = unpack(snapshot.encoded, snapshot.edges);
  const { encoded, edges, ...stats } = snapshot;
  return { ...stats, deployBytes: encoded.length, scores: data.map(events => evaluate(model, events, { details })) };
}

function compression(reference, referenceBytes, measured, measuredBytes, n) {
  const predictionSavingPerEvent = reference - measured;
  const extraModelBits = (measuredBytes - referenceBytes) * 8;
  return { predictionSavingPerEvent, extraModelBits, netExtraBits: extraModelBits - n * predictionSavingPerEvent,
    breakEvenEvents: predictionSavingPerEvent > 0 ? Math.ceil(Math.max(0, extraModelBits) / predictionSavingPerEvent) : null };
}

function blockStudy(model, edges, data, seed, memory, baseScores) {
  const mature = edges.filter(edge => edge.state === 'mature');
  const groups = memory ? [0, 1] : [0];
  let projections;
  const baseline = baseScores.map(score => score.bits);
  return groups.map(regime => {
    // Select by training metadata alone. Retain complete batches and require a
    // disjoint, same-size control on every draw, within a 2x magnitude ratio.
    const idsByBatch = new Map();
    for (const edge of mature) {
      if (!idsByBatch.has(edge.batch)) idsByBatch.set(edge.batch, []);
      idsByBatch.get(edge.batch).push(edge);
    }
    const eligible = [...idsByBatch.values()].filter(group => !memory || group[0].trainingRegime === regime);
    if (!eligible.length) return { regime, status: 'no-mature-batch' };
    const attempts = [];
    for (const selected of eligible.slice(0, 5)) {
      const target = new Set(selected.map(edge => edge.id));
      const pool = edges.filter(edge => !target.has(edge.id));
      const controls = [];
      for (let draw = 0; draw < 20; draw++) {
        const rng = stream(seed, 411, regime, draw);
        const used = new Set();
        for (const edge of selected.slice().sort((a, b) => Math.abs(b.w) - Math.abs(a.w))) {
          const weight = Math.max(1e-5, Math.abs(edge.w));
          const eligibleEdges = pool.filter(candidate => {
            const candidateWeight = Math.max(1e-5, Math.abs(candidate.w));
            return !used.has(candidate.id) && candidate.backbone === edge.backbone
              && (candidate.i === candidate.j) === (edge.i === edge.j)
              && candidateWeight >= weight / 2 && candidateWeight <= weight * 2;
          });
          if (!eligibleEdges.length) break;
          const nonMature = eligibleEdges.filter(candidate => candidate.state !== 'mature');
          const choices = nonMature.length ? nonMature : eligibleEdges;
          used.add(rng.choice(choices).id);
        }
        if (used.size !== target.size) break;
        controls.push(used);
      }
      if (controls.length !== 20) continue;
      projections ??= data.map(events => events.map(event =>
        model.project(event.features, event.target + FIRST - HORIZONS[0])));
      const delta = blocked => data.map((events, k) =>
        evaluate(model, events, { blocked, projections: projections[k] }).bits - baseline[k]);
      attempts.push({ batch: selected[0].batch, edges: [...target], deltaBits: delta(target),
        controls: controls.map(blocked => ({ edges: [...blocked], deltaBits: delta(blocked) })),
        restoredDeltaBits: delta(new Set()) });
      break;
    }
    return { regime, status: attempts.length ? 'matched' : 'insufficient-matches', results: attempts };
  });
}

function auditState(network) {
  const edges = network.edges.map(({ m, v, ...edge }) => edge);
  const batches = network.batches.map(({ edges, ...batch }) => ({ ...batch, edges: edges.map(edge => edge.id) }));
  return { edges, batches, proposals: network.proposals };
}

function trainingState(network) {
  const array = values => Array.from(values);
  const { edges, batches, proposals } = auditState(network);
  return {
    config: network.config, events: network.events, work: network.work, nextId: network.nextId,
    input: array(network.input), output: array(network.output), bias: array(network.bias), gain: array(network.gain),
    activity: array(network.activity), feedback: array(network.feedback),
    optimizer: network.optimizer.map(({ m, v }) => ({ m: array(m), v: array(v) })),
    liveEdges: network.edges.filter(edge => edge.deleted === null),
    window: network.window.map(({ a, b }) => ({ a: a.map(array), b: b.map(array) })),
    liveBatches: batches.filter(batch => batch.state !== 'deleted').map(({ windows, ...batch }) => batch),
    auditBytes: Buffer.byteLength(JSON.stringify({ edges, batches, proposals }), 'utf8'),
  };
}

function optionsFor(method, config) {
  if (method === 'local_fixed' || method === 'local_random' || method === 'local_joint') {
    return { ...config, mode: method.slice(6), learner: 'local' };
  }
  if (method === 'joint_no_prune') return { ...config, mode: 'joint', prune: false };
  if (method === 'joint_no_stability') return { ...config, mode: 'joint', matureRate: 1 };
  if (method === 'joint_no_homeostasis') return { ...config, mode: 'joint', homeostasis: false };
  return { ...config, mode: method };
}

function runSeed(config, seed, dir, log) {
  const data = evaluationData(config, seed);
  const methods = Object.fromEntries(config.methods.map(method => [method, new SynapticNetwork(optionsFor(method, { ...config, seed }))]));
  const points = new Map();
  const counts = new Uint32Array(CLASSES).fill(1);
  const uniform = evaluate(null, data[0], { prior: frequencies(new Uint32Array(CLASSES).fill(1)) });
  const memoryTimeline = [];
  let trained = 0;
  function record() {
    const details = trained === sections(config).length * config.train && config.profile === 'formal';
    points.set(trained, { prior: data.map(events => evaluate(null, events, { prior: frequencies(counts) })),
      methods: Object.fromEntries(Object.entries(methods).map(([name, network]) => [name, scoreSnapshot(network.snapshot(), data, details)])) });
  }
  record();
  for (const section of sections(config)) {
    for (let index = 0; index < section.events; index++) {
      const event = makeEvent(seed, 12, trained, config.encoder, section.regime, config.memory);
      for (const network of Object.values(methods)) network.train(event);
      counts[event.feedback]++;
      trained++;
      if (config.checkpoints.includes(trained) || index === section.events - 1 ||
          (config.memory && section.name === 'A2' && (index + 1) % config.interval === 0)) {
        record();
      }
    }
    if (config.memory) memoryTimeline.push({ phase: section.name, checkpoint: trained });
  }
  const latest = points.get(trained);
  const finals = {};
  for (const [method, network] of Object.entries(methods)) {
    const snapshot = network.snapshot();
    const filename = path.join(dir, `${method}.syn`);
    fs.writeFileSync(filename, snapshot.encoded, { flag: 'wx' });
    for (const edge of snapshot.edges) edge.trainingRegime = network.batches.find(batch => batch.id === edge.batch)?.regime;
    const model = unpack(fs.readFileSync(filename), snapshot.edges);
    const readback = data.map(events => evaluate(model, events));
    if (readback.some((score, k) => score.bits !== latest.methods[method].scores[k].bits)) {
      throw new Error(`Stored model differs from scored model: ${method}`);
    }
    const audit = auditState(network);
    const state = trainingState(network);
    if (GROWTH.has(network.mode)) fs.writeFileSync(path.join(dir, `${method}-edges.json`), JSON.stringify(audit), { flag: 'wx' });
    finals[method] = {
      model: path.basename(filename), trainingStateBytes: Buffer.byteLength(JSON.stringify(state)), auditBytes: state.auditBytes,
      shuffledBits: evaluate(model, data[0], { pairing: pairPermutation(data[0].length, seed) }).bits,
      compressionVsPrior: compression(latest.prior[0].bits, counts.byteLength, readback[0].bits, snapshot.encoded.length, data[0].length),
      ...(method !== 'random' && latest.methods.random ? { compressionVsRandom: compression(latest.methods.random.scores[0].bits,
        latest.methods.random.deployBytes, readback[0].bits, snapshot.encoded.length, data[0].length) } : {}),
      blocking: method === 'joint' ? blockStudy(model, snapshot.edges, data, seed, config.memory, readback) : undefined,
    };
  }
  // The uniform prior has no learned bytes. The frequency prior stores 25 uint32 counts.
  const priorBuffer = Buffer.alloc(counts.byteLength);
  counts.forEach((count, k) => priorBuffer.writeUInt32LE(count, k * 4));
  const priorFile = path.join(dir, 'prior.bin');
  fs.writeFileSync(priorFile, priorBuffer, { flag: 'wx' });
  const priorReadback = fs.readFileSync(priorFile);
  const storedPrior = frequencies(Array.from({ length: CLASSES }, (_, k) => priorReadback.readUInt32LE(4 * k)));
  if (evaluate(null, data[0], { prior: storedPrior }).bits !== latest.prior[0].bits) throw new Error('Stored prior differs from scored prior');
  const compute = config.computeMatch ? computeMatched(config, seed, data, methods) : undefined;
  const memory = config.memory ? memoryResults(config, seed, data, points, memoryTimeline) : undefined;
  const physics = data.map((events, regime) => {
    const split = { pilot: 20, development: 21, formal: 93 }[config.profile];
    return events.filter((event, i) => physicalOracle(sceneAt(seed, split, i, regime, config.memory)) === event.target + FIRST).length / events.length;
  });
  log(`seed=${seed} ${Object.entries(latest.methods).map(([method, row]) => `${method}=${row.scores[0].bits.toFixed(4)}`).join(' ')}`);
  return { seed, uniform, physics, checkpoints: Object.fromEntries(points), latest: trained, final: finals, compute, memory };
}

function replay(config, seed, method, total, policy) {
  const network = new SynapticNetwork(optionsFor(method, { ...config, seed }));
  for (let index = 0; index < total; index++) {
    const { regime, identity = index } = policy(index);
    network.train(makeEvent(seed, 12, identity, config.encoder, regime, config.memory));
  }
  return network;
}

function computeMatched(config, seed, data, methods) {
  const budget = Math.min(...Object.values(methods).map(network => network.work));
  const matched = {};
  for (const [method, original] of Object.entries(methods)) {
    const overBudget = original.workHistory.findIndex(work => work > budget);
    const count = overBudget < 0 ? original.events : overBudget - 1;
    const network = count === original.events ? original : replay(config, seed, method, count, i => ({
      regime: config.memory && Math.floor(i / config.train) === 1 ? 1 : 0,
    }));
    if (network.work !== original.workHistory[count]) throw new Error('Compute budget replay diverged');
    matched[method] = scoreSnapshot(network.snapshot(), data);
  }
  return { budget, matched };
}

function memoryResults(config, seed, data, points, timeline) {
  const [a, b, end] = timeline.map(phase => points.get(phase.checkpoint));
  const summary = {};
  for (const method of config.methods) {
    const before = a.methods[method].scores[0].bits;
    const recovered = [...points].find(([step, row]) => step > config.train * 2 && row.methods[method].scores[0].bits <= before + 0.05);
    const constantA = replay(config, seed, method, config.train * 3, () => ({ regime: 0 }));
    const constantB = replay(config, seed, method, config.train * 2, () => ({ regime: 1 }));
    const freshA = replay(config, seed, method, config.train, i => ({ regime: 0, identity: config.train * 2 + i }));
    summary[method] = { forgettingBits: b.methods[method].scores[0].bits - before,
      bAfterB: b.methods[method].scores[1].bits, aAfterReturn: end.methods[method].scores[0].bits,
      recoveredByEvents: recovered ? recovered[0] - config.train * 2 : null,
      constantA: scoreSnapshot(constantA.snapshot(), data).scores[0].bits,
      constantB: scoreSnapshot(constantB.snapshot(), data).scores[1].bits,
      freshA: scoreSnapshot(freshA.snapshot(), data).scores[0].bits };
  }
  return { timeline, methods: summary };
}

function paired(values, seed, interval = 0.95) {
  const rng = stream(seed, 902);
  const draws = Array.from({ length: 10000 }, () => mean(values.map(() => rng.choice(values)))).sort((a, b) => a - b);
  const tail = (1 - interval) / 2;
  return { perSeed: values, mean: mean(values), interval, bounds: [draws[Math.floor(tail * (draws.length - 1))],
    draws[Math.floor((1 - tail) * (draws.length - 1))]] };
}

function aggregate(reports, config) {
  const latest = report => report.checkpoints[report.latest];
  const fixed = reports.filter(report => latest(report).methods.fixed);
  const developmentGate = fixed.length ? {
    vsUniform: paired(fixed.map(report => report.uniform.bits - latest(report).methods.fixed.scores[0].bits), config.seed),
    vsFrequency: paired(fixed.map(report => latest(report).prior[0].bits - latest(report).methods.fixed.scores[0].bits), config.seed),
    matchedVsShuffled: paired(fixed.map(report => report.final.fixed.shuffledBits - latest(report).methods.fixed.scores[0].bits), config.seed),
  } : null;
  if (developmentGate) developmentGate.passed = developmentGate.vsUniform.mean >= 0.1 && developmentGate.vsFrequency.mean >= 0.1
    && developmentGate.matchedVsShuffled.bounds[0] > 0;
  const compare = (control, trial, level = 0.95) => config.methods.includes(control) && config.methods.includes(trial)
    ? paired(reports.map(report => latest(report).methods[control].scores[0].bits - latest(report).methods[trial].scores[0].bits), config.seed, level)
    : null;
  const primary = compare('random', 'joint');
  return { developmentGate, primary, primaryThresholdMet: primary ? primary.mean >= 0.05 && primary.bounds[0] > 0 : null,
    jointVsHebb: compare('hebb', 'joint', 0.975), jointVsBidirectional: compare('bidirectional', 'joint', 0.975),
    jointVsShuffled: compare('shuffled', 'joint'), localJointVsRandom: compare('local_random', 'local_joint') };
}

function codeHash() {
  const hash = createHash('sha256');
  for (const file of ['synaptic_scene.js', 'synaptic_network.js', 'synaptic_experiment.js', 'predictive_growth_experiment.js']) {
    hash.update(fs.readFileSync(path.join(__dirname, file)));
  }
  return hash.digest('hex');
}

function prepareStudy(config, output, workers) {
  validate(config);
  fs.mkdirSync(output, { recursive: true });
  // Reject accidental overwrites before any files are created.
  if (fs.readdirSync(output).length) throw new Error('Output directory must be empty');
  const locked = { config, codeHash: codeHash(), workers, machine: { node: process.version, platform: process.platform,
    cpu: os.cpus()[0]?.model, memoryBytes: os.totalmem() } };
  fs.writeFileSync(path.join(output, 'protocol.json'), JSON.stringify(locked, null, 2), { flag: 'wx' });
  return locked;
}

function studySeeds(config) {
  return Array.from({ length: config.seeds }, (_, i) => i === 0 ? config.seed : stream(config.seed, 198, i).state);
}

function runStudySeed(config, output, seed, log) {
  const dir = path.join(output, `seed-${seed}`);
  fs.mkdirSync(dir);
  const report = runSeed(config, seed, dir, log);
  fs.writeFileSync(path.join(dir, 'report.json'), JSON.stringify(report, null, config.profile === 'formal' ? 0 : 2), { flag: 'wx' });
  return report;
}

function finishStudy(locked, output, reports) {
  const { config } = locked;
  const summary = { ...locked, ...aggregate(reports, config), seeds: reports.map(report => {
    const last = report.checkpoints[report.latest];
    return { seed: report.seed, uniform: report.uniform.bits, prior: last.prior[0].bits,
      methods: Object.fromEntries(Object.entries(last.methods).map(([name, row]) => [name, { bits: row.scores[0].bits,
        anticipation: row.scores[0].anticipation, maxAhead: row.scores[0].maxAhead, activeEdges: row.activeEdges, mature: row.mature,
        deployBytes: row.deployBytes, compressionVsPrior: report.final[name].compressionVsPrior }])) };
  }) };
  fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(summary, null, 2), { flag: 'wx' });
  return summary;
}

function runStudy(config, output, log = console.log) {
  const locked = prepareStudy(config, output, 1);
  const reports = studySeeds(config).map(seed => runStudySeed(config, output, seed, log));
  return finishStudy(locked, output, reports);
}

async function runParallelStudy(config, output, workers) {
  workers = Math.min(workers, config.seeds);
  const locked = prepareStudy(config, output, workers);
  const seeds = studySeeds(config);
  // The parent is the sole writer of protocol.json and summary.json.
  // Each worker owns a disjoint subset of seed directories.
  await Promise.all(Array.from({ length: workers }, (_, index) => new Promise((resolve, reject) => {
    const assigned = seeds.filter((_, i) => i % workers === index);
    const worker = new Worker(__filename, { workerData: { config, output, seeds: assigned } });
    worker.once('error', reject);
    worker.once('exit', code => code === 0 ? resolve() : reject(new Error(`Worker exited with code ${code}`)));
  })));
  const reports = seeds.map(seed => JSON.parse(fs.readFileSync(path.join(output, `seed-${seed}`, 'report.json'), 'utf8')));
  return finishStudy(locked, output, reports);
}

function validate(config) {
  for (const key of ['seed', 'seeds', 'train', 'samples', 'interval', 'candidates', 'edgesPerGrowth', 'maxEdges', 'maxIn', 'fixedCapacity', 'confirmEvents', 'confirmWindows']) {
    if (!Number.isSafeInteger(config[key]) || config[key] < 1) throw new Error(`Expected positive integer: ${key}`);
  }
  if (config.seed > 0xffffffff || config.samples < 2 || config.maxEdges > 1024 || config.maxEdges < 128 ||
      config.fixedCapacity < 128 || config.fixedCapacity > config.maxEdges || config.interval < 2 || config.maxIn > 32) {
    throw new Error('Invalid seed, sample count, or network limits');
  }
  if (!['pilot', 'development', 'formal'].includes(config.profile) || !['motion', 'pooled'].includes(config.encoder) ||
      !['bptt', 'local'].includes(config.learner) || !Number.isFinite(config.rate) || config.rate <= 0 ||
      !Number.isFinite(config.traceDecay) || config.traceDecay < 0 || config.traceDecay > 1 ||
      !Number.isFinite(config.support) || config.support < 0 || !Number.isFinite(config.matureRate) || config.matureRate < 0 || config.matureRate > 1) {
    throw new Error('Invalid profile, encoder, or training hyperparameters');
  }
  if (!config.methods.length || new Set(config.methods).size !== config.methods.length ||
      config.methods.some(method => !METHODS.includes(method) && !['local_fixed', 'local_random', 'local_joint',
        'joint_no_prune', 'joint_no_stability', 'joint_no_homeostasis'].includes(method))) throw new Error('Invalid method list');
  if (config.checkpoints.some(n => !Number.isSafeInteger(n) || n < 0 || n > config.train * (config.memory ? 3 : 1))) {
    throw new Error('Invalid checkpoint list');
  }
  if (config.profile === 'formal' && (config.seeds < 10 || config.samples < 2000)) throw new Error('Formal profile requires at least 10 seeds and 2000 test events');
  if (config.memory && config.train % config.interval !== 0) throw new Error('Align phase boundaries with the growth interval');
}

async function main() {
  const { values } = parseArgs({ options: {
    profile: { type: 'string' }, seed: { type: 'string' }, seeds: { type: 'string' }, train: { type: 'string' },
    samples: { type: 'string' }, checkpoints: { type: 'string' }, rate: { type: 'string' }, encoder: { type: 'string' },
    learner: { type: 'string' }, 'trace-decay': { type: 'string' }, methods: { type: 'string' },
    'fixed-capacity': { type: 'string' }, 'compute-match': { type: 'boolean' }, memory: { type: 'boolean' },
    output: { type: 'string' }, help: { type: 'boolean' }, workers: { type: 'string' },
  } });
  if (values.help) {
    console.log('node synaptic_experiment.js --profile pilot|development|formal --train 1024 --seeds 3 --samples 256 --output synaptic-pilot\n'
      + '--encoder motion|pooled --learner bptt|local --trace-decay 1 --methods fixed,random,hebb,bidirectional,joint,shuffled,capacity\n'
      + '--compute-match --memory --workers 1 are optional. --train is events per phase in memory mode.');
    return;
  }
  const profile = values.profile ?? 'pilot';
  const train = Number(values.train ?? (profile === 'formal' || profile === 'development' ? 8192 : defaults.train));
  const config = { ...defaults, profile, train,
    seeds: Number(values.seeds ?? (profile === 'formal' ? 10 : defaults.seeds)),
    samples: Number(values.samples ?? (profile === 'formal' ? 2000 : defaults.samples)),
    checkpoints: values.checkpoints ? values.checkpoints.split(',').map(Number) : [...new Set([0, 128, 512, 2048, train])].filter(n => n <= train),
  };
  for (const key of ['seed', 'rate', 'encoder', 'learner', 'memory']) {
    if (key in values) config[key] = ['seed', 'rate'].includes(key) ? Number(values[key]) : values[key];
  }
  if (values.methods) config.methods = values.methods.split(',').map(method => method.trim());
  if ('trace-decay' in values) config.traceDecay = Number(values['trace-decay']);
  if ('fixed-capacity' in values) config.fixedCapacity = Number(values['fixed-capacity']);
  if ('compute-match' in values) config.computeMatch = values['compute-match'];
  const output = path.resolve(values.output ?? `synaptic-${profile}`);
  const workers = Number(values.workers ?? 1);
  if (!Number.isSafeInteger(workers) || workers < 1) throw new Error('Expected positive integer: workers');
  const summary = workers > 1 ? await runParallelStudy(config, output, workers) : runStudy(config, output);
  console.log(JSON.stringify({ output, developmentGate: summary.developmentGate?.passed, primary: summary.primary }, null, 2));
}

if (require.main === module) {
  if (isMainThread) {
    main().catch(error => { console.error(error.message); process.exitCode = 1; });
  } else {
    const { config, output, seeds } = workerData;
    for (const seed of seeds) runStudySeed(config, output, seed, console.log);
  }
}

module.exports = { defaults, evaluate, paired, runStudy };
