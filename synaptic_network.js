'use strict';

const { CLASSES, featureCount, stream, shuffle } = require('./synaptic_scene');

const NODES = 32;
const INITIAL_EDGES = 128;
const LEAK = 0.45;
const LN2 = Math.LN2;
const GROWTH = new Set(['random', 'hebb', 'bidirectional', 'joint', 'shuffled']);

function distribution(logits) {
  const max = Math.max(...logits);
  const p = Float64Array.from(logits, value => Math.exp(value - max));
  const sum = p.reduce((a, b) => a + b, 0);
  for (let c = 0; c < p.length; c++) p[c] /= sum;
  return p;
}

function loss(p, target) {
  return -Math.log(Math.max(p[target], Number.MIN_VALUE));
}

function initialPairs(seed) {
  const pairs = [];
  for (let j = 0; j < NODES; j++) {
    pairs.push({ i: (j + NODES - 1) % NODES, j, backbone: true }, { i: (j + 1) % NODES, j, backbone: true });
  }
  const remaining = [];
  for (let j = 0; j < NODES; j++) {
    for (let i = 0; i < NODES; i++) {
      if (i !== (j + NODES - 1) % NODES && i !== (j + 1) % NODES) remaining.push({ i, j, backbone: false });
    }
  }
  return [...pairs, ...shuffle(remaining, stream(seed, 102))];
}

function ranks(items, key) {
  const ordered = items.slice().sort((a, b) => a[key] - b[key]);
  for (let start = 0; start < ordered.length;) {
    let end = start + 1;
    while (end < ordered.length && ordered[start][key] === ordered[end][key]) end++;
    const percentile = (start + end) / (2 * ordered.length);
    for (let k = start; k < end; k++) ordered[k][`${key}Rank`] = percentile;
    start = end;
  }
}

function evidence(window, pair, permuted) {
  let q = 0, shuffledQ = 0;
  let x = 0, y = 0, xx = 0, yy = 0, xy = 0;
  let eventsWithTransitions = 0;
  for (let k = 0; k < window.length; k++) {
    const { a, b } = window[k];
    const shuffledB = window[permuted[k]].b;
    if (a.length <= 1) continue;
    eventsWithTransitions++;
    const scale = 1 / (a.length - 1);
    for (let t = 1; t < a.length; t++) {
      const source = a[t - 1][pair.i], target = a[t][pair.j];
      q -= source * b[t][pair.j];
      shuffledQ -= source * (shuffledB[t]?.[pair.j] ?? 0);
      x += source * scale;
      y += target * scale;
      xx += source ** 2 * scale;
      yy += target ** 2 * scale;
      xy += source * target * scale;
    }
  }
  const n = eventsWithTransitions || 1;
  const spread = Math.sqrt(Math.max(0, (xx - x * x / n) * (yy - y * y / n)));
  const hebb = spread > 1e-10 ? Math.max(0, (xy - x * y / n) / spread) : 0;
  return { ...pair, q: q / window.length, absQ: Math.abs(q / window.length),
    shuffledQ: shuffledQ / window.length, absShuffled: Math.abs(shuffledQ / window.length), hebb };
}

class SynapticNetwork {
  constructor(config) {
    this.config = config;
    this.mode = config.mode;
    this.inputs = featureCount(config.encoder);
    const rng = stream(config.seed, 100);
    this.input = Float64Array.from({ length: NODES * this.inputs }, (_, k) => {
      return rng.normal() * (k % this.inputs > 256 ? 0.22 : 0.075);
    });
    this.output = Float64Array.from({ length: CLASSES * NODES }, () => rng.normal() * 0.14);
    this.bias = new Float64Array(CLASSES);
    this.gain = new Float64Array(NODES).fill(1);
    this.activity = new Float64Array(NODES).fill(0.045);
    const feedback = stream(config.seed, 101);
    this.feedback = Float64Array.from({ length: CLASSES * NODES }, () => feedback.normal() * 0.14);
    this.pairs = initialPairs(config.seed);
    this.edges = [];
    this.batches = [];
    this.proposals = [];
    this.window = [];
    this.events = 0;
    this.work = 0;
    this.workHistory = [0];
    this.nextId = 0;
    const start = this.mode === 'capacity' ? config.fixedCapacity : INITIAL_EDGES;
    const weight = stream(config.seed, 103);
    for (const pair of this.pairs.slice(0, start)) this.addEdge(pair, weight.normal() * 0.1, null);
    this.optimizer = [this.input, this.output, this.bias].map(weights => ({
      m: new Float64Array(weights.length), v: new Float64Array(weights.length),
    }));
    this.peakRss = process.memoryUsage.rss();
    this.trainMs = 0;
    this.maintenanceMs = 0;
  }

  addEdge({ i, j, backbone }, w, batch) {
    const edge = { id: this.nextId++, i, j, w, birthWeight: w, backbone, born: this.events,
      deleted: null, state: batch ? 'trial' : 'initial', batch, m: 0, v: 0 };
    this.edges.push(edge);
    return edge;
  }

  project(features, stop) {
    const projected = [];
    for (let t = 0; t <= stop; t++) {
      const sums = new Float64Array(NODES);
      const frame = features[t];
      for (let j = 0; j < NODES; j++) {
        const offset = j * this.inputs;
        let sum = 0;
        for (let k = 0; k < this.inputs; k++) sum += this.input[offset + k] * frame[k];
        sums[j] = sum;
      }
      projected.push(sums);
    }
    return projected;
  }

  forward(features, stop, blocked, countWork = false, inputDrive) {
    const edges = this.edges.filter(edge => edge.deleted === null && !blocked?.has(edge.id));
    const a = [new Float64Array(NODES)];
    const activation = [];
    const projection = inputDrive ?? this.project(features, stop);
    for (let t = 0; t <= stop; t++) {
      const sums = projection[t].slice();
      for (const edge of edges) sums[edge.j] += edge.w * a[t][edge.i];
      const nonlinearity = sums.map((sum, j) => Math.tanh(this.gain[j] * sum));
      activation.push(nonlinearity);
      a.push(nonlinearity.map((value, j) => LEAK * a[t][j] + (1 - LEAK) * value));
    }
    if (countWork) this.work += (stop + 1) * ((inputDrive ? 0 : NODES * this.inputs) + edges.length + NODES);
    return { a, activation, edges, inputDrive: projection };
  }

  read(a) {
    const logits = this.bias.slice();
    for (let c = 0; c < CLASSES; c++) {
      for (let j = 0; j < NODES; j++) logits[c] += this.output[c * NODES + j] * a[j];
    }
    return distribution(logits);
  }

  gradient(features, prefixes, target, precomputed, countWork = false) {
    const stop = prefixes.at(-1);
    const { a, activation, edges } = precomputed ?? this.forward(features, stop, null, countWork);
    const inGrad = new Float64Array(this.input.length), outGrad = new Float64Array(this.output.length);
    const biasGrad = new Float64Array(CLASSES), edgeGrad = new Float64Array(edges.length);
    const delta = Array.from({ length: stop + 1 }, () => new Float64Array(NODES));
    const b = Array.from({ length: stop + 1 }, () => new Float64Array(NODES));
    const local = this.config.learner === 'local';
    const feedback = local ? this.feedback : this.output;
    let eventLoss = 0;
    for (const t of prefixes) {
      const p = this.read(a[t + 1]);
      eventLoss += loss(p, target) / prefixes.length;
      for (let c = 0; c < CLASSES; c++) {
        const gradient = (p[c] - Number(c === target)) / prefixes.length;
        const offset = c * NODES;
        biasGrad[c] += gradient;
        for (let j = 0; j < NODES; j++) {
          outGrad[offset + j] += gradient * a[t + 1][j];
          delta[t][j] += gradient * feedback[offset + j];
        }
      }
    }
    for (let t = stop; t >= 0; t--) {
      for (let j = 0; j < NODES; j++) {
        const derivative = (1 - LEAK) * this.gain[j] * (1 - activation[t][j] ** 2);
        const signal = delta[t][j] * derivative;
        b[t][j] = signal;
        if (t > 0) delta[t - 1][j] += (local ? this.config.traceDecay : 1) * LEAK * delta[t][j];
        const offset = j * this.inputs;
        for (let k = 0; k < this.inputs; k++) inGrad[offset + k] += signal * features[t][k];
      }
      for (let k = 0; k < edges.length; k++) {
        const { i, j, w } = edges[k];
        edgeGrad[k] += b[t][j] * a[t][i];
        // Local propagation retains the target unit's own Jacobian only. This reverse
        // evaluation is equivalent to its scalar forward eligibility trace.
        if (t > 0 && (!local || i === j)) {
          delta[t - 1][i] += (local ? this.config.traceDecay : 1) * w * b[t][j];
        }
      }
    }
    if (countWork) this.work += (stop + 1) * (NODES * this.inputs + 2 * edges.length + 2 * NODES)
      + prefixes.length * 3 * CLASSES * NODES;
    return { inGrad, outGrad, biasGrad, edgeGrad, edges, a: a.slice(1), b, eventLoss };
  }

  apply(gradient) {
    const step = this.events, rate = this.config.rate;
    const arrays = [this.input, this.output, this.bias];
    const grads = [gradient.inGrad, gradient.outGrad, gradient.biasGrad];
    const c1 = 1 - 0.9 ** step, c2 = 1 - 0.999 ** step;
    for (let k = 0; k < arrays.length; k++) {
      const values = arrays[k], g = grads[k], { m, v } = this.optimizer[k];
      for (let i = 0; i < values.length; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        v[i] = 0.999 * v[i] + 0.001 * g[i] ** 2;
        values[i] -= rate * (m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8);
      }
      this.work += values.length * 8;
    }
    gradient.edges.forEach((edge, k) => {
      const g = gradient.edgeGrad[k];
      edge.m = 0.9 * edge.m + 0.1 * g;
      edge.v = 0.999 * edge.v + 0.001 * g ** 2;
      const age = step - edge.born;
      const scale = edge.state === 'mature' ? this.config.matureRate : 1;
      edge.w -= rate * scale * (edge.m / (1 - 0.9 ** age)) / (Math.sqrt(edge.v / (1 - 0.999 ** age)) + 1e-8);
    });
    this.work += 8 * gradient.edges.length;
  }

  observeBatches(event, baseLoss, inputDrive) {
    for (const batch of this.batches) {
      if (batch.state === 'deleted' || this.events - batch.born < this.config.interval) continue;
      const blocked = new Set(batch.edges.map(edge => edge.id));
      const alternate = this.forward(event.features, event.prefixes.at(-1), blocked, true, inputDrive);
      const lossBlocked = event.prefixes.reduce((sum, t) => sum + loss(this.read(alternate.a[t + 1]), event.feedback), 0) / event.prefixes.length;
      this.work += event.prefixes.length * CLASSES * NODES;
      batch.sum += (lossBlocked - baseLoss) / LN2;
      batch.samples++;
      if (batch.samples < this.config.confirmEvents) continue;
      const gain = batch.sum / batch.samples;
      batch.windows.push({ through: this.events + 1, gainBits: gain });
      batch.samples = 0;
      batch.sum = 0;
      batch.positive = gain > this.config.support ? batch.positive + 1 : 0;
      batch.negative = gain < this.config.support ? batch.negative + 1 : 0;
      if (batch.positive >= this.config.confirmWindows && batch.state !== 'mature') {
        batch.state = 'mature';
        batch.matured = this.events + 1;
        batch.edges.forEach(edge => { edge.state = 'mature'; });
      }
      if (batch.negative >= this.config.confirmWindows && this.config.prune) {
        batch.state = 'deleted';
        for (const edge of batch.edges) {
          edge.state = 'deleted';
          edge.deleted = this.events + 1;
        }
      }
    }
  }

  propose(regime) {
    const rng = stream(this.config.seed, 104, this.events);
    const pool = [], indegree = new Uint16Array(NODES);
    const used = new Set();
    for (const edge of this.edges) {
      if (edge.deleted === null) {
        used.add(edge.i * NODES + edge.j);
        indegree[edge.j]++;
      }
    }
    const vacancies = Math.min(this.config.edgesPerGrowth, this.config.maxEdges - used.size);
    if (vacancies <= 0) {
      this.proposals.push({ event: this.events, skipped: 'capacity' });
      return;
    }
    for (const pair of this.pairs) {
      if (indegree[pair.j] < this.config.maxIn && !used.has(pair.i * NODES + pair.j)) pool.push(pair);
    }
    shuffle(pool, rng);
    const candidates = pool.slice(0, this.config.candidates);
    if (candidates.length === 0) {
      this.proposals.push({ event: this.events, skipped: 'indegree' });
      return;
    }
    const n = this.window.length;
    const shuffled = shuffle(Array.from({ length: n }, (_, k) => k), rng);
    const permuted = [];
    shuffled.forEach((k, idx) => { permuted[k] = shuffled[(idx + 1) % n]; });
    const scored = candidates.map(pair => evidence(this.window, pair, permuted));
    const transitions = this.window.reduce((sum, event) => sum + event.a.length - 1, 0);
    this.work += candidates.length * (transitions * 12 + 12 * Math.ceil(Math.log2(candidates.length + 1)));
    for (const key of ['absQ', 'hebb', 'absShuffled']) ranks(scored, key);
    for (const candidate of scored) {
      candidate.joint = candidate.absQRank * (0.5 + 0.5 * candidate.hebbRank);
      candidate.shuffled = candidate.absShuffledRank * (0.5 + 0.5 * candidate.hebbRank);
    }
    const key = { hebb: 'hebb', bidirectional: 'absQ', joint: 'joint', shuffled: 'shuffled' }[this.mode];
    const ordered = scored.slice();
    if (key) ordered.sort((a, b) => b[key] - a[key]);
    const batch = { id: this.batches.length + 1, born: this.events, regime, state: 'trial',
      edges: [], windows: [], samples: 0, sum: 0, positive: 0, negative: 0, matured: null };
    const selected = [];
    for (let slot = 0; slot < vacancies; slot++) {
      const explore = slot === vacancies - 1;
      const from = explore ? scored : ordered;
      const next = from.find(edge => !selected.includes(edge) && indegree[edge.j] < this.config.maxIn);
      if (!next) continue;
      const direction = Math.sign(next.q) || (rng.next() < 0.5 ? -1 : 1);
      batch.edges.push(this.addEdge(next, 0.015 * direction, batch.id));
      next.selected = explore ? 'explore' : 'rank';
      selected.push(next);
      indegree[next.j]++;
    }
    if (batch.edges.length) this.batches.push(batch);
    this.proposals.push({ event: this.events, candidates: scored });
  }

  train(event) {
    const start = performance.now();
    // The forward pass uses only selected, pre-impact frames. Sound onset is
    // revealed afterwards and supplies the sole training target.
    const saved = this.forward(event.features, event.prefixes.at(-1), null, true);
    const gradient = this.gradient(event.features, event.prefixes, event.feedback, saved, true);
    const maintenance = performance.now();
    if (GROWTH.has(this.mode)) {
      // Baseline and blocked passes share the input weights until apply().
      this.observeBatches(event, gradient.eventLoss, saved.inputDrive);
      this.window.push({ a: gradient.a, b: gradient.b });
    }
    const update = performance.now();
    this.events++;
    this.apply(gradient);
    if (this.config.homeostasis) {
      for (let j = 0; j < NODES; j++) {
        const power = gradient.a.reduce((sum, a) => sum + a[j] ** 2, 0) / gradient.a.length;
        this.activity[j] = 0.995 * this.activity[j] + 0.005 * power;
        const adjustment = Math.max(-0.0004, Math.min(0.0004, 0.004 * (0.045 - this.activity[j])));
        this.gain[j] = Math.min(2, Math.max(0.5, this.gain[j] * (1 + adjustment)));
      }
      this.work += NODES * (gradient.a.length * 2 + 8);
    }
    const structure = performance.now();
    if (this.events % this.config.interval === 0) {
      if (GROWTH.has(this.mode)) {
        this.propose(event.regime);
        this.window.length = 0;
      }
      this.peakRss = Math.max(this.peakRss, process.memoryUsage.rss());
    }
    this.workHistory.push(this.work);
    this.maintenanceMs += update - maintenance + performance.now() - structure;
    this.trainMs += maintenance - start + structure - update;
  }

  snapshot() {
    const live = this.edges.filter(edge => edge.deleted === null);
    const encoded = pack(this.config.encoder, this.input, this.output, this.bias, this.gain, live);
    return { encoded, edges: live.map(edge => ({ ...edge })), events: this.events, work: this.work,
      mature: live.filter(edge => edge.state === 'mature').length, activeEdges: live.length,
      rssBytes: Math.max(this.peakRss, process.memoryUsage.rss()), trainMs: this.trainMs,
      maintenanceMs: this.maintenanceMs };
  }
}

// Header (magic, encoder, shape), float32 arrays, and compact uint16 edge endpoints.
function pack(encoder, input, output, bias, gain, edges) {
  const buffer = Buffer.alloc(12 + 4 * (input.length + output.length + bias.length + gain.length) + 8 * edges.length);
  buffer.write('SYN1');
  buffer.writeUInt16LE(encoder === 'motion' ? 1 : 0, 4);
  buffer.writeUInt16LE(NODES, 6);
  buffer.writeUInt16LE(CLASSES, 8);
  buffer.writeUInt16LE(edges.length, 10);
  let offset = 12;
  for (const values of [input, output, bias, gain]) {
    for (const value of values) { buffer.writeFloatLE(value, offset); offset += 4; }
  }
  for (const edge of edges) {
    buffer.writeUInt16LE(edge.i, offset);
    buffer.writeUInt16LE(edge.j, offset + 2);
    buffer.writeFloatLE(edge.w, offset + 4);
    offset += 8;
  }
  return buffer;
}

function unpack(buffer, metadata) {
  if (buffer.length < 12 || buffer.toString('ascii', 0, 4) !== 'SYN1'
    || buffer.readUInt16LE(4) > 1 || buffer.readUInt16LE(6) !== NODES || buffer.readUInt16LE(8) !== CLASSES) {
    throw new Error('Invalid synaptic model header');
  }
  const encoder = buffer.readUInt16LE(4) ? 'motion' : 'pooled';
  const m = Object.create(SynapticNetwork.prototype);
  m.inputs = featureCount(encoder);
  m.config = { encoder };
  const edgeCount = buffer.readUInt16LE(10);
  const expected = 12 + 4 * (m.inputs * NODES + NODES * CLASSES + CLASSES + NODES) + 8 * edgeCount;
  if (buffer.length !== expected || (metadata && metadata.length !== edgeCount)) throw new Error('Invalid synaptic model size');
  let offset = 12;
  const floats = n => Float64Array.from({ length: n }, () => {
    const value = buffer.readFloatLE(offset);
    offset += 4;
    if (!Number.isFinite(value)) throw new Error('Invalid model parameter');
    return value;
  });
  m.input = floats(m.inputs * NODES);
  m.output = floats(NODES * CLASSES);
  m.bias = floats(CLASSES);
  m.gain = floats(NODES);
  m.edges = Array.from({ length: edgeCount }, (_, k) => {
    const edge = { ...metadata?.[k], deleted: null, id: metadata?.[k]?.id ?? k,
      i: buffer.readUInt16LE(offset), j: buffer.readUInt16LE(offset + 2), w: buffer.readFloatLE(offset + 4) };
    offset += 8;
    if (edge.i >= NODES || edge.j >= NODES || !Number.isFinite(edge.w)) throw new Error('Invalid synaptic edge');
    return edge;
  });
  return m;
}

module.exports = { NODES, INITIAL_EDGES, LN2, GROWTH, SynapticNetwork, distribution, loss, evidence, unpack };
