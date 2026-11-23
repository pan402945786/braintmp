#!/usr/bin/env node

/**
 * Predictive growth experiment, v0.1.
 *
 * This is intentionally dependency free so the experiment can be run on a
 * fresh Node installation.  It implements the first-round design in
 * predictive_growth_experiment.md: a 48-frame visual stream, a synchronous
 * impact sound, online learning after each event, dynamic sparse connections,
 * random/experience-selected growth controls, and the requested metrics.
 */

'use strict';

const fs = require('fs');

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function mean(xs) { return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0; }
function median(xs) {
  if (!xs.length) return NaN;
  const a = xs.slice().sort((x, y) => x - y);
  const m = Math.floor(a.length / 2);
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}
function log2(x) { return Math.log(x) / Math.LN2; }

class RNG {
  constructor(seed = 1) { this.state = (seed >>> 0) || 1; }
  next() {
    let x = this.state;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 0x100000000;
  }
  uniform(a = 0, b = 1) { return a + (b - a) * this.next(); }
  int(a, b) { return Math.floor(this.uniform(a, b + 1)); }
  choice(xs) { return xs[this.int(0, xs.length - 1)]; }
  normal() {
    const u = Math.max(this.next(), 1e-12), v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}

class SceneGenerator {
  constructor(seed = 1, cfg = {}) {
    this.rng = new RNG(seed);
    this.frames = cfg.frames || 48;
    this.width = cfg.width || 32;
    this.height = cfg.height || 32;
    this.tMin = cfg.tMin || 16;
    this.tMax = cfg.tMax || 40;
  }

  event(id = 0) {
    const T = this.rng.int(this.tMin, this.tMax);
    const radius = this.rng.uniform(1.8, 3.2);
    const x0 = this.rng.uniform(7, 25);
    const vx = this.rng.uniform(-0.24, 0.24);
    const bg = this.rng.uniform(0.03, 0.12);
    const appearance = this.rng.uniform(0.75, 1.25);
    // Choose y0/vy so that the first contact is exactly T under constant g.
    const g = this.rng.uniform(0.075, 0.105);
    const yContact = this.rng.uniform(25.5, 28.0);
    const y0 = this.rng.uniform(4.0, 11.0);
    const vy = (yContact - y0 - 0.5 * g * T * T) / T;
    const frames = new Array(this.frames);
    const audio = new Float32Array(this.frames);
    for (let t = 0; t < this.frames; t++) {
      const pixels = new Float32Array(this.width * this.height);
      for (let i = 0; i < pixels.length; i++) {
        // Deterministic, weak background texture independent of T.
        const n = ((i * 1103515245 + id * 12345 + t * 2654435761) >>> 0) / 0x100000000;
        pixels[i] = bg + 0.03 * (n - 0.5);
      }
      const x = x0 + vx * t;
      const y = y0 + vy * t + 0.5 * g * t * t;
      const sigma = radius * 0.72;
      for (let py = Math.max(0, Math.floor(y - 4 * radius)); py <= Math.min(this.height - 1, Math.ceil(y + 4 * radius)); py++) {
        for (let px = Math.max(0, Math.floor(x - 4 * radius)); px <= Math.min(this.width - 1, Math.ceil(x + 4 * radius)); px++) {
          const d2 = (px - x) ** 2 + (py - y) ** 2;
          const value = appearance * Math.exp(-d2 / (2 * sigma * sigma));
          pixels[py * this.width + px] = clamp(pixels[py * this.width + px] + value, 0, 1);
        }
      }
      frames[t] = pixels;
      if (t === T) audio[t] = 1;
      else if (t === T + 1 && t < this.frames) audio[t] = 0.35;
    }
    return { id, T, frames, audio, meta: { x0, vx, y0, vy, g, radius } };
  }

  batch(n, startId = 0) {
    const out = [];
    for (let i = 0; i < n; i++) out.push(this.event(startId + i));
    return out;
  }
}

function encodeFrame(event, t) {
  const pixels = event.frames[t];
  let sum = 0, sx = 0, sy = 0, max = 0;
  const side = Math.sqrt(pixels.length);
  for (let i = 0; i < pixels.length; i++) {
    const v = pixels[i]; sum += v; if (v > max) max = v;
    sx += (i % side) * v; sy += Math.floor(i / side) * v;
  }
  const mass = Math.max(sum, 1e-6);
  const cx = sx / mass / side, cy = sy / mass / side;
  const audio = event.audio[t];
  const prevAudio = t ? event.audio[t - 1] : 0;
  return new Float64Array([cx, cy, clamp(sum / pixels.length * 8, 0, 1), max, audio, audio - prevAudio]);
}

function softmax(logits) {
  let mx = -Infinity; for (const z of logits) if (z > mx) mx = z;
  const out = new Float64Array(logits.length); let s = 0;
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - mx); s += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= s || 1;
  return out;
}

function weightedMedian(probs, tMin) {
  let c = 0; for (let i = 0; i < probs.length; i++) { c += probs[i]; if (c >= 0.5) return tMin + i; }
  return tMin + probs.length - 1;
}

class SparseGrowingRNN {
  constructor(cfg = {}) {
    this.tMin = cfg.tMin || 16; this.tMax = cfg.tMax || 40;
    this.outputDim = this.tMax - this.tMin + 1;
    this.inputDim = cfg.inputDim || 6;
    this.hiddenCapacity = cfg.hiddenCapacity || 24;
    this.activeHidden = cfg.activeHidden || 4;
    this.learningRate = cfg.learningRate || 0.035;
    this.growthInterval = cfg.growthInterval || 64;
    this.edgesPerGrowth = cfg.edgesPerGrowth || 3;
    this.maintenanceWindow = cfg.maintenanceWindow || 64;
    this.stabilityThreshold = cfg.stabilityThreshold || 0.0009;
    this.growthMode = cfg.growthMode || 'experience';
    this.rng = new RNG(cfg.seed || 1);
    this.edges = [];
    this.out = Array.from({ length: this.hiddenCapacity }, () => new Float64Array(this.outputDim));
    this.outBias = new Float64Array(this.outputDim);
    this.hiddenBias = new Float64Array(this.hiddenCapacity);
    this.recentScores = new Map();
    this.recentLosses = [];
    this.eventCount = 0;
    this._initEdges(cfg.fullCapacity === true);
  }

  sourceCount() { return this.inputDim + this.activeHidden; }
  _initEdges(full) {
    const hidden = full ? this.hiddenCapacity : this.activeHidden;
    if (full) this.activeHidden = this.hiddenCapacity;
    for (let d = 0; d < hidden; d++) {
      for (let s = 0; s < this.inputDim + hidden; s++) {
        if (!full && s >= this.inputDim && s - this.inputDim >= this.activeHidden) continue;
        this._addEdge(s, d, 0, 0, true);
      }
      for (let c = 0; c < this.outputDim; c++) this.out[d][c] = this.rng.normal() * 0.08;
    }
    for (let c = 0; c < this.outputDim; c++) this.outBias[c] = 0;
  }

  _edgeKey(s, d) { return `${s}:${d}`; }
  _addEdge(source, dest, weight, birthEvent, initial = false) {
    const key = this._edgeKey(source, dest);
    if (this.edges.some(e => e.key === key)) return null;
    const e = { key, source, dest, weight: weight || (this.rng.normal() * 0.04), birthEvent: this.eventCount, deleteEvent: null, stable: false, contribution: 0, uses: 0 };
    if (!initial && birthEvent != null) e.birthEvent = birthEvent;
    this.edges.push(e); return e;
  }

  clone() {
    const c = new SparseGrowingRNN({ tMin: this.tMin, tMax: this.tMax, inputDim: this.inputDim, hiddenCapacity: this.hiddenCapacity, activeHidden: this.activeHidden, learningRate: this.learningRate, growthInterval: this.growthInterval, edgesPerGrowth: this.edgesPerGrowth, maintenanceWindow: this.maintenanceWindow, stabilityThreshold: this.stabilityThreshold, growthMode: this.growthMode, seed: 1 });
    c.edges = this.edges.map(e => ({ ...e }));
    c.out = this.out.map(row => Float64Array.from(row)); c.outBias = Float64Array.from(this.outBias); c.hiddenBias = Float64Array.from(this.hiddenBias);
    c.eventCount = this.eventCount; c.recentScores = new Map(this.recentScores); c.recentLosses = this.recentLosses.slice();
    return c;
  }

  _edgeMap() { const m = new Map(); for (const e of this.edges) if (e.deleteEvent == null) m.set(e.key, e); return m; }
  forwardFeatures(features, prev = null, blocked = null) {
    const h = new Float64Array(this.hiddenCapacity); const prevH = prev || new Float64Array(this.hiddenCapacity);
    const em = this._edgeMap();
    for (let d = 0; d < this.activeHidden; d++) {
      let z = this.hiddenBias[d];
      for (let s = 0; s < this.inputDim + this.activeHidden; s++) {
        const e = em.get(this._edgeKey(s, d));
        if (!e || (blocked && blocked.has(e.key))) continue;
        const v = s < this.inputDim ? features[s] : prevH[s - this.inputDim];
        z += e.weight * v;
      }
      h[d] = Math.tanh(z);
    }
    const logits = Float64Array.from(this.outBias);
    for (let d = 0; d < this.activeHidden; d++) for (let c = 0; c < this.outputDim; c++) logits[c] += h[d] * this.out[d][c];
    return { h, logits, probs: softmax(logits) };
  }

  predict(event, cutoff, blocked = null) {
    let h = new Float64Array(this.hiddenCapacity), state = null;
    for (let t = 0; t <= cutoff; t++) { state = this.forwardFeatures(encodeFrame(event, t), h, blocked); h = state.h; }
    return state.probs;
  }

  _recordScore(source, dest, value) {
    const key = this._edgeKey(source, dest);
    this.recentScores.set(key, (this.recentScores.get(key) || 0) * 0.95 + Math.abs(value) * 0.05);
  }

  trainEvent(event, prefixCount = 3) {
    const states = []; let h = new Float64Array(this.hiddenCapacity);
    for (let t = 0; t < event.T; t++) { const s = this.forwardFeatures(encodeFrame(event, t), h); states.push({ ...s, features: encodeFrame(event, t), prev: h }); h = s.h; }
    const prefixes = [];
    for (let k = 1; k <= prefixCount; k++) prefixes.push(Math.min(event.T - 1, 3 + Math.floor(k * 8 / prefixCount)));
    let eventLoss = 0;
    for (const p of prefixes) {
      const st = states[p]; const grad = Float64Array.from(st.probs); const target = event.T - this.tMin;
      eventLoss += -Math.log(Math.max(st.probs[target], 1e-12)); grad[target] -= 1;
      const dh = new Float64Array(this.activeHidden);
      for (let d = 0; d < this.activeHidden; d++) {
        let g = 0; for (let c = 0; c < this.outputDim; c++) g += grad[c] * this.out[d][c]; dh[d] = g * (1 - st.h[d] * st.h[d]);
        for (let c = 0; c < this.outputDim; c++) this.out[d][c] -= this.learningRate * grad[c] * st.h[d];
      }
      for (let c = 0; c < this.outputDim; c++) this.outBias[c] -= this.learningRate * grad[c];
      const em = this._edgeMap();
      for (let d = 0; d < this.activeHidden; d++) for (let s = 0; s < this.inputDim + this.activeHidden; s++) {
        const e = em.get(this._edgeKey(s, d)); if (!e) continue;
        const src = s < this.inputDim ? st.features[s] : st.prev[s - this.inputDim];
        const g = dh[d] * src; e.weight -= this.learningRate * g; e.contribution = 0.98 * e.contribution + 0.02 * Math.abs(g * e.weight); e.uses++;
        this._recordScore(s, d, g);
      }
      for (const e of this.edges) if (e.dest < dh.length) e.contribution = 0.995 * e.contribution + 0.005 * Math.abs(dh[e.dest]);
    }
    this.recentLosses.push(eventLoss / prefixes.length); if (this.recentLosses.length > this.maintenanceWindow) this.recentLosses.shift();
    this.eventCount++;
    if (this.eventCount % this.growthInterval === 0) this.grow();
    if (this.eventCount % this.maintenanceWindow === 0) this.maintain();
    return eventLoss / prefixes.length;
  }

  grow() {
    const candidates = [];
    const used = new Set(this.edges.map(e => e.key));
    for (let d = 0; d < this.activeHidden; d++) for (let s = 0; s < this.inputDim + this.activeHidden; s++) {
      const key = this._edgeKey(s, d); if (!used.has(key)) candidates.push({ s, d, score: this.recentScores.get(key) || 0 });
    }
    if (this.activeHidden < this.hiddenCapacity) {
      const d = this.activeHidden++;
      for (let s = 0; s < this.inputDim; s++) candidates.push({ s, d, score: this.recentScores.get(this._edgeKey(s, d)) || 0 });
      for (let c = 0; c < this.outputDim; c++) this.out[d][c] = this.rng.normal() * 0.08;
    }
    if (this.growthMode === 'random') {
      for (let i = candidates.length - 1; i > 0; i--) { const j = this.rng.int(0, i); [candidates[i], candidates[j]] = [candidates[j], candidates[i]]; }
    } else candidates.sort((a, b) => b.score - a.score);
    let added = 0; for (const c of candidates) { if (added >= this.edgesPerGrowth) break; if (this._addEdge(c.s, c.d, this.rng.normal() * 0.04, this.eventCount)) added++; }
  }

  maintain() {
    for (const e of this.edges) {
      if (this.eventCount - e.birthEvent >= this.maintenanceWindow && e.contribution >= this.stabilityThreshold) e.stable = true;
    }
    const removable = this.edges.filter(e => !e.stable && this.eventCount - e.birthEvent >= this.maintenanceWindow && e.contribution < this.stabilityThreshold);
    for (const e of removable.slice(0, Math.max(0, this.edges.length - this.inputDim * 2))) e.deleteEvent = this.eventCount;
    this.learningRate *= 0.995;
  }

  matureEdges() { return this.edges.filter(e => e.deleteEvent == null && e.stable).sort((a, b) => b.contribution - a.contribution); }
  blockedClone(fraction = 1, random = false) {
    const c = this.clone(); const edges = c.matureEdges(); const n = Math.max(1, Math.floor(edges.length * fraction));
    const chosen = random ? c.edges.filter(e => e.deleteEvent == null && !e.stable).sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, n) : edges.slice(0, n);
    return { model: c, blocked: new Set(chosen.map(e => e.key)) };
  }

  serialize(precision = 6) {
    const q = x => Number(Number(x).toFixed(precision));
    return JSON.stringify({ schema: 'predictive-growth-rnn/v1', precision, tMin: this.tMin, tMax: this.tMax, inputDim: this.inputDim, hiddenCapacity: this.hiddenCapacity, activeHidden: this.activeHidden, eventCount: this.eventCount,
      edges: this.edges.map(e => ({ ...e, weight: q(e.weight), contribution: q(e.contribution) })), out: this.out.map(row => Array.from(row, q)), outBias: Array.from(this.outBias, q), hiddenBias: Array.from(this.hiddenBias, q) });
  }
  static deserialize(serialized) {
    const r = typeof serialized === 'string' ? JSON.parse(serialized) : serialized;
    const m = new SparseGrowingRNN({ tMin: r.tMin, tMax: r.tMax, inputDim: r.inputDim, hiddenCapacity: r.hiddenCapacity, activeHidden: r.activeHidden, seed: 1 });
    m.edges = r.edges.map(e => ({ ...e })); m.out = r.out.map(row => Float64Array.from(row)); m.outBias = Float64Array.from(r.outBias); m.hiddenBias = Float64Array.from(r.hiddenBias); m.eventCount = r.eventCount || 0;
    return m;
  }
  stats() {
    return { activeHidden: this.activeHidden, activeEdges: this.edges.filter(e => e.deleteEvent == null).length, stableEdges: this.edges.filter(e => e.deleteEvent == null && e.stable).length, deletedEdges: this.edges.filter(e => e.deleteEvent != null).length, modelBits: Buffer.byteLength(this.serialize(), 'utf8') * 8, events: this.eventCount };
  }
}

function priorFromEvents(events, tMin, tMax) {
  const counts = new Float64Array(tMax - tMin + 1); for (const e of events) counts[e.T - tMin]++;
  for (let i = 0; i < counts.length; i++) counts[i] += 1;
  const z = counts.reduce((a, b) => a + b, 0); return { probs: Float64Array.from(counts, x => x / z), serialize: () => JSON.stringify(Array.from(counts)), stats: () => ({ modelBits: Buffer.byteLength(JSON.stringify(Array.from(counts)), 'utf8') * 8, activeEdges: 0, stableEdges: 0, activeHidden: 0 }) };
}

function evaluate(model, events, cutoffs, cfg = {}) {
  const tMin = cfg.tMin || 16, tMax = cfg.tMax || 40;
  const losses = [], absErrors = [];
  for (let i = 0; i < events.length; i++) {
    const e = events[i], p = model.probs ? model.probs : model.predict(e, cutoffs[i], cfg.blocked || null);
    const target = e.T - tMin; losses.push(-log2(Math.max(p[target], 1e-12))); absErrors.push(Math.abs(weightedMedian(p, tMin) - e.T));
  }
  const ahead = [2, 4, 6, 8, 10, 12], curve = {};
  for (const a of ahead) {
    let ok = 0; for (const e of events) { const cutoff = Math.max(0, e.T - a); const p = model.probs ? model.probs : model.predict(e, cutoff, cfg.blocked || null); if (Math.abs(weightedMedian(p, tMin) - e.T) <= 1) ok++; }
    curve[a] = ok / events.length;
  }
  let maxAhead = 0; for (const a of ahead) { if (curve[a] >= 0.8) maxAhead = a; else break; }
  return { n: events.length, nllBits: mean(losses), medianAbsFrameError: median(absErrors), anticipationAccuracy: curve, maxContinuousAheadFrames: maxAhead };
}

function evaluatePrior(prior, events, cutoffs, cfg) {
  return evaluate({ probs: prior.probs }, events, cutoffs, cfg);
}

function runExperiment(options = {}) {
  const cfg = {
    seed: options.seed || 7, trainEvents: options.trainEvents || 512, devEvents: options.devEvents || 128, testEvents: options.testEvents || 256,
    checkpoints: options.checkpoints || [0, 128, 512], tMin: 16, tMax: 40,
    hiddenCapacity: options.hiddenCapacity || 24, initialHidden: options.initialHidden || 4,
  };
  cfg.checkpoints = cfg.checkpoints.filter(x => x <= cfg.trainEvents);
  if (!cfg.checkpoints.includes(0)) cfg.checkpoints.unshift(0);
  const train = new SceneGenerator(cfg.seed, cfg).batch(cfg.trainEvents, 0);
  const dev = new SceneGenerator(cfg.seed + 1, cfg).batch(cfg.devEvents, 100000);
  const test = new SceneGenerator(cfg.seed + 2, cfg).batch(cfg.testEvents, 200000);
  const crng = new RNG(cfg.seed + 3), cutoffs = test.map(() => crng.int(4, 12));
  const drng = new RNG(cfg.seed + 4), devCutoffs = dev.map(() => drng.int(4, 12));
  const methods = {
    fixed_initial: new SparseGrowingRNN({ ...cfg, activeHidden: cfg.initialHidden, hiddenCapacity: cfg.hiddenCapacity, growthMode: 'none', seed: cfg.seed + 10, growthInterval: Number.MAX_SAFE_INTEGER }),
    random_growth: new SparseGrowingRNN({ ...cfg, activeHidden: cfg.initialHidden, hiddenCapacity: cfg.hiddenCapacity, growthMode: 'random', seed: cfg.seed + 11 }),
    experience_growth: new SparseGrowingRNN({ ...cfg, activeHidden: cfg.initialHidden, hiddenCapacity: cfg.hiddenCapacity, growthMode: 'experience', seed: cfg.seed + 12 }),
    fixed_capacity: new SparseGrowingRNN({ ...cfg, activeHidden: cfg.hiddenCapacity, hiddenCapacity: cfg.hiddenCapacity, fullCapacity: true, growthMode: 'none', seed: cfg.seed + 13, growthInterval: Number.MAX_SAFE_INTEGER }),
  };
  const snapshots = { time_prior: {} }; for (const cp of cfg.checkpoints) snapshots.time_prior[cp] = priorFromEvents(train.slice(0, cp), cfg.tMin, cfg.tMax);
  const snap = { fixed_initial: {}, random_growth: {}, experience_growth: {}, fixed_capacity: {} };
  for (const name of Object.keys(methods)) snap[name][0] = methods[name].clone();
  for (let i = 0; i < train.length; i++) {
    for (const model of Object.values(methods)) model.trainEvent(train[i]);
    const done = i + 1; if (cfg.checkpoints.includes(done)) for (const [name, model] of Object.entries(methods)) snap[name][done] = model.clone();
  }
  const results = {}, development = {};
  for (const cp of cfg.checkpoints) {
    results[cp] = {}; development[cp] = {};
    const prior = snapshots.time_prior[cp]; results[cp].time_prior = { metrics: evaluatePrior(prior, test, cutoffs, cfg), stats: prior.stats() }; development[cp].time_prior = evaluatePrior(prior, dev, devCutoffs, cfg);
    for (const [name] of Object.entries(methods)) {
      // Score the quantized, serialized representation so compression and
      // prediction refer to the same model artifact.
      const model = SparseGrowingRNN.deserialize(snap[name][cp].serialize()); const metrics = evaluate(model, test, cutoffs, cfg);
      development[cp][name] = evaluate(model, dev, devCutoffs, cfg);
      const blocked = model.blockedClone(1, false); const blockedMetrics = evaluate(blocked.model, test, cutoffs, { ...cfg, blocked: blocked.blocked });
      const randomBlocked = model.blockedClone(1, true); const randomBlockedMetrics = evaluate(randomBlocked.model, test, cutoffs, { ...cfg, blocked: randomBlocked.blocked });
      results[cp][name] = { metrics, stats: model.stats(), blocking: { matureDeltaBits: blockedMetrics.nllBits - metrics.nllBits, randomDeltaBits: randomBlockedMetrics.nllBits - metrics.nllBits, matureEdgesBlocked: blocked.blocked.size } };
    }
  }
  const baseBits = results[cfg.checkpoints[0]].time_prior.stats.modelBits;
  for (const cp of cfg.checkpoints) for (const [name, row] of Object.entries(results[cp])) {
    row.compression = { modelBits: row.stats.modelBits, netBitsVsPrior: (row.metrics.nllBits - results[cp].time_prior.metrics.nllBits) * cfg.testEvents + row.stats.modelBits - baseBits, estimatedBreakEvenEvents: row.metrics.nllBits < results[cp].time_prior.metrics.nllBits ? Math.max(0, (row.stats.modelBits - baseBits) / (results[cp].time_prior.metrics.nllBits - row.metrics.nllBits)) : null };
  }
  return { config: cfg, results, development, notes: ['All methods use identical train/dev/test seeds and per-event cutoffs.', 'The visual encoder reads only pixels/audio up to the requested cutoff.', 'Use larger train/test counts for the formal run; defaults are a smoke-test scale.'] };
}

function bootstrapCI(values, seed = 1, draws = 1000) {
  if (!values.length) return [NaN, NaN];
  const rng = new RNG(seed), means = [];
  for (let b = 0; b < draws; b++) { let total = 0; for (let i = 0; i < values.length; i++) total += values[rng.int(0, values.length - 1)]; means.push(total / values.length); }
  means.sort((a, b) => a - b); return [means[Math.floor(0.025 * (means.length - 1))], means[Math.floor(0.975 * (means.length - 1))]];
}

/** Run paired independent seeds and return seed-level means plus bootstrap CIs. */
function runStudy(options = {}) {
  const count = options.seeds || 10, baseSeed = options.seed || 7, reports = [];
  for (let i = 0; i < count; i++) reports.push(runExperiment({ ...options, seed: baseSeed + i, seeds: undefined }));
  const checkpoints = reports[0].config.checkpoints, aggregate = {};
  for (const cp of checkpoints) {
    aggregate[cp] = {};
    const names = Object.keys(reports[0].results[cp]);
    for (const name of names) {
      const vals = reports.map(r => r.results[cp][name].metrics.nllBits);
      const ci = bootstrapCI(vals, baseSeed + cp + name.length);
      aggregate[cp][name] = { meanNllBits: mean(vals), ci95NllBits: ci, n: vals.length };
    }
    if (names.includes('random_growth') && names.includes('experience_growth')) {
      const vals = reports.map(r => r.results[cp].random_growth.metrics.nllBits - r.results[cp].experience_growth.metrics.nllBits);
      aggregate[cp].experienceGrowthDeltaVsRandom = { mean: mean(vals), ci95: bootstrapCI(vals, baseSeed + cp + 101) };
    }
  }
  return { seeds: reports.map(r => r.config.seed), aggregate, reports };
}

function parseArgs(argv) {
  const out = {}; for (let i = 0; i < argv.length; i++) { const a = argv[i]; if (!a.startsWith('--')) continue; const k = a.slice(2).replace(/-([a-z])/g, (_, c) => c.toUpperCase()); const v = argv[i + 1]; out[k] = v && !v.startsWith('--') ? (/^\d+$/.test(v) ? Number(v) : v) : true; if (out[k] !== true) i++; }
  if (out.checkpoints) out.checkpoints = String(out.checkpoints).split(',').map(Number);
  return out;
}

if (require.main === module) {
  const args = parseArgs(process.argv.slice(2)); const result = runExperiment(args);
  const output = args.output || 'predictive_growth_results.json'; fs.writeFileSync(output, JSON.stringify(result, null, 2));
  const summary = Object.fromEntries(Object.entries(result.results).map(([cp, rows]) => [cp, Object.fromEntries(Object.entries(rows).map(([name, row]) => [name, Number(row.metrics.nllBits.toFixed(4))]))]));
  console.log(`Wrote ${output}`); console.log(JSON.stringify(summary, null, 2));
}

module.exports = { RNG, SceneGenerator, SparseGrowingRNN, priorFromEvents, evaluate, runExperiment, runStudy, bootstrapCI, encodeFrame, weightedMedian };

