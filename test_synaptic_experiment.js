'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const { FIRST, LAST, FRAMES, FLOOR, RADIUS, sceneAt, height, physicalOracle, render, encode, makeEvent } = require('./synaptic_scene');
const { SynapticNetwork, NODES, loss, evidence, unpack } = require('./synaptic_network');
const { defaults, evaluate } = require('./synaptic_experiment');

const config = { ...defaults, seed: 7, mode: 'joint' };
const close = (actual, expected, tol = 1e-7) => assert.ok(Math.abs(actual - expected) < tol, `${actual} != ${expected}`);

test('contact, feedback and prefix rendering agree across both gravity settings', () => {
  for (const regime of [0, 1]) {
    const seen = new Set();
    for (let i = 0; i < 220; i++) {
      const scene = sceneAt(7, 10, i, regime, true);
      seen.add(scene.contact);
      assert.equal(physicalOracle(scene), scene.contact);
      assert.ok(scene.contact >= FIRST && scene.contact <= LAST);
      for (let t = 0; t < scene.contact; t++) {
        assert.ok(height(scene, t) >= RADIUS && height(scene, t) + RADIUS < FLOOR);
      }
    }
    assert.equal(seen.size, LAST - FIRST + 1);
    const scene = sceneAt(7, 10, 5, regime, true);
    const full = render(scene);
    const prefix = render(scene, 8);
    assert.equal(full.length, FRAMES);
    assert.equal(full.findIndex(frame => frame.audio > 0), scene.contact);
    assert.deepEqual(full.slice(0, 9), prefix);
    for (const encoder of ['pooled', 'motion']) assert.deepEqual(encode(full, encoder).slice(0, 9), encode(prefix, encoder));
  }
});

test('time gradients agree with finite differences including a dormant edge', () => {
  const model = new SynapticNetwork(config);
  const event = makeEvent(91, 12, 0, config.encoder);
  const { features, prefixes, feedback } = event;
  const value = () => {
    const { a } = model.forward(features, prefixes.at(-1));
    return prefixes.reduce((s, t) => s + loss(model.read(a[t + 1]), feedback), 0) / prefixes.length;
  };
  const finite = (get, set) => {
    const w = get(), epsilon = 1e-5;
    set(w + epsilon);
    const plus = value();
    set(w - epsilon);
    const minus = value();
    set(w);
    return (plus - minus) / (2 * epsilon);
  };
  const gradients = model.gradient(features, prefixes, feedback);
  for (const [param, deriv, index] of [[model.input, gradients.inGrad, 257],
    [model.output, gradients.outGrad, feedback * NODES + 1], [model.bias, gradients.biasGrad, feedback]]) {
    assert.ok(Math.abs(deriv[index]) > 1e-8);
    close(deriv[index], finite(() => param[index], w => { param[index] = w; }));
  }
  const edge = model.edges[0];
  close(gradients.edgeGrad[0], finite(() => edge.w, w => { edge.w = w; }));
  const missing = model.pairs.find(pair => !model.edges.some(e => e.i === pair.i && e.j === pair.j));
  const q = evidence([gradients], missing, [0]).q;
  assert.ok(Math.abs(q) > 1e-8);
  const added = model.addEdge(missing, 0, 1);
  close(-q, finite(() => added.w, w => { added.w = w; }));
});

test('local feedback agrees with an explicit decaying forward eligibility trace', () => {
  const model = new SynapticNetwork({ ...config, learner: 'local', traceDecay: 0.6 });
  const event = makeEvent(10, 12, 1, config.encoder);
  const edge = model.edges[0];
  const pass = model.forward(event.features, event.prefixes.at(-1));
  const grad = model.gradient(event.features, event.prefixes, event.feedback, pass);
  const selfWeight = model.edges.find(e => e.i === edge.j && e.j === edge.j)?.w ?? 0;
  let trace = 0;
  let expected = 0;
  for (let t = 0; t <= event.prefixes.at(-1); t++) {
    const derivative = 0.55 * model.gain[edge.j] * (1 - pass.activation[t][edge.j] ** 2);
    trace = derivative * pass.a[t][edge.i] + model.config.traceDecay * (0.45 + selfWeight * derivative) * trace;
    if (!event.prefixes.includes(t)) continue;
    const p = model.read(pass.a[t + 1]);
    const signal = p.reduce((sum, value, c) => sum + (value - Number(c === event.feedback)) * model.feedback[c * NODES + edge.j], 0);
    expected += signal * trace / event.prefixes.length;
  }
  close(grad.edgeGrad[0], expected);
});

test('identical starting networks score missing edges and trial edges respect the resource budget', () => {
  const joint = new SynapticNetwork({ ...config, maxEdges: 132, fixedCapacity: 132 });
  const random = new SynapticNetwork({ ...joint.config, mode: 'random' });
  assert.deepEqual(joint.snapshot().encoded, random.snapshot().encoded);
  for (let i = 0; i < joint.config.interval; i++) {
    const event = makeEvent(7, 12, i, config.encoder);
    joint.train(event);
    random.train(event);
  }
  const round = joint.proposals[0];
  assert.equal(round.candidates.length, 128);
  assert.ok(round.candidates.some(candidate => Math.abs(candidate.q) > 1e-7));
  assert.deepEqual(round.candidates.map(c => [c.i, c.j, c.q]), random.proposals[0].candidates.map(c => [c.i, c.j, c.q]));
  assert.equal(joint.snapshot().activeEdges, 132);
  assert.equal(joint.batches[0].edges.length, 4);
  assert.equal(round.candidates.filter(c => c.selected === 'explore').length, 1);
  assert.equal(new Set(joint.edges.map(e => `${e.i}:${e.j}`)).size, joint.edges.length);
  assert.ok(joint.edges.every(e => e.i < NODES && e.j < NODES));
  const snapshot = joint.snapshot();
  const model = unpack(snapshot.encoded, snapshot.edges);
  const data = [makeEvent(7, 20, 0, config.encoder), makeEvent(7, 20, 1, config.encoder)];
  const base = evaluate(model, data);
  const blocked = new Set(joint.batches[0].edges.map(e => e.id));
  assert.notEqual(evaluate(model, data, { blocked }).bits, base.bits);
  assert.deepEqual(evaluate(model, data), base);
  assert.throws(() => unpack(snapshot.encoded.subarray(0, -1)), /size/);
});

test('reusing the input projection preserves blocked predictions and counts shared work once', () => {
  const model = new SynapticNetwork(config);
  const data = [makeEvent(7, 20, 0, config.encoder), makeEvent(7, 20, 1, config.encoder)];
  const { features, prefixes } = data[0];
  const stop = prefixes.at(-1);
  const base = model.forward(features, stop, null, true);
  const blocked = new Set([model.edges[0].id]);
  const workBefore = model.work;
  const cached = model.forward(features, stop, blocked, true, base.inputDrive);
  assert.equal(model.work - workBefore, (stop + 1) * (cached.edges.length + NODES));
  assert.deepEqual(cached, model.forward(features, stop, blocked));
  const projections = data.map(event => model.project(event.features, event.target + FIRST - 2));
  assert.deepEqual(evaluate(model, data, { blocked, projections }), evaluate(model, data, { blocked }));
});

test('trial confirmation can mature, retire and reuse a connection', () => {
  const model = new SynapticNetwork({ ...config, interval: 2, confirmEvents: 1, confirmWindows: 2 });
  model.events = 10;
  const pair = model.pairs.find(p => !model.edges.some(edge => edge.i === p.i && edge.j === p.j));
  const edge = model.addEdge(pair, 0.01, 1);
  model.batches.push({ id: 1, born: model.events, state: 'trial', edges: [edge],
    windows: [], samples: 0, sum: 0, positive: 0, negative: 0 });
  const event = makeEvent(7, 12, 4, config.encoder);
  model.observeBatches(event, 0);
  assert.equal(edge.state, 'trial');
  assert.equal(model.batches[0].windows.length, 0);
  model.events += 2;
  model.observeBatches(event, 0);
  model.events++;
  model.observeBatches(event, 0);
  assert.equal(edge.state, 'mature');
  model.events++;
  model.observeBatches(event, 100);
  model.events++;
  model.observeBatches(event, 100);
  assert.equal(edge.state, 'deleted');
  assert.equal(model.snapshot().activeEdges, 128);
  const reformed = model.addEdge(pair, 0.02, 2);
  assert.notEqual(reformed.id, edge.id);
});

test('hierarchical features are causal, local, differentiable and serializable', () => {
  const empty = () => ({ pixels: new Float32Array(1024), audio: 0 });
  const a = empty(), b = empty();
  b.pixels[8 * 32 + 8] = 1;
  const temporal = encode([a, b], 'hierarchical-delta');
  assert.equal(temporal[0].length, 385);
  assert.ok(temporal[0].slice(320, 384).every(v => v === 0));
  assert.ok(temporal[1].slice(320, 384).some(v => v > 0));
  assert.equal(temporal[1][63], 0);
  for (const encoder of ['hierarchical', 'hierarchical-delta']) {
    const scene = sceneAt(7, 10, 5);
    assert.deepEqual(encode(render(scene), encoder).slice(0, 9), encode(render(scene, 8), encoder));
    const model = new SynapticNetwork({ ...config, encoder, mode: 'fixed' });
    const loaded = unpack(model.snapshot().encoded);
    const event = makeEvent(7, 12, 3, encoder), stop = event.prefixes.at(-1);
    const original = model.forward(event.features, stop).a[stop + 1];
    const restored = loaded.forward(event.features, stop).a[stop + 1];
    original.forEach((v, k) => close(v, restored[k], 1e-6));
    const gradients = model.gradient(event.features, event.prefixes, event.feedback);
    const index = gradients.inGrad.findIndex(x => Math.abs(x) > 1e-7);
    const saved = model.input[index], eps = 1e-5;
    const objective = () => {
      const states = model.forward(event.features, stop).a;
      return event.prefixes.reduce((sum, t) => sum + loss(model.read(states[t + 1]), event.feedback), 0) / event.prefixes.length;
    };
    model.input[index] = saved + eps; const plus = objective();
    model.input[index] = saved - eps; const minus = objective();
    model.input[index] = saved;
    close(gradients.inGrad[index], (plus - minus) / (2 * eps));
  }
});

test('learned local motion and delayed feedback have correct gradients and round trip', () => {
  for (const encoder of ['hierarchical-motion', 'hierarchical-feedback']) {
    const model = new SynapticNetwork({ ...config, encoder, mode: 'fixed' });
    const event = makeEvent(7, 12, 3, encoder), stop = event.prefixes.at(-1);
    const pass = model.forward(event.features, stop);
    const gradients = model.gradient(event.features, event.prefixes, event.feedback, pass);
    const objective = () => {
      const states = model.forward(event.features, stop).a;
      return event.prefixes.reduce((sum, t) => sum + loss(model.read(states[t + 1]), event.feedback), 0) / event.prefixes.length;
    };
    const checks = [[model.input, gradients.inGrad], [model.motionKernel, gradients.motionGrad], [model.output, gradients.outGrad], [model.bias, gradients.biasGrad]];
    if (encoder.endsWith('feedback')) checks.push([model.visualFeedback, gradients.visualFeedbackGrad]);
    for (const [values, grad] of checks) {
      const index = grad.reduce((best, x, i) => Math.abs(x) > Math.abs(grad[best]) ? i : best, 0);
      assert.ok(Math.abs(grad[index]) > 1e-10);
      const saved = values[index], eps = 1e-5;
      values[index] = saved + eps; const plus = objective();
      values[index] = saved - eps; const minus = objective();
      values[index] = saved;
      close(grad[index], (plus - minus) / (2 * eps), 1e-7);
    }
    const edge = model.edges[0], old = edge.w, eps = 1e-5;
    edge.w = old + eps; const plus = objective(); edge.w = old - eps; const minus = objective(); edge.w = old;
    close(gradients.edgeGrad[0], (plus - minus) / (2 * eps));
    const missing = model.pairs.find(pair => !model.edges.some(e => e.i === pair.i && e.j === pair.j));
    const candidate = model.addEdge(missing, 0, 1);
    candidate.w = eps; const candidatePlus = objective();
    candidate.w = -eps; const candidateMinus = objective();
    candidate.w = 0;
    close(-evidence([gradients], missing, [0]).q, (candidatePlus - candidateMinus) / (2 * eps));
    model.edges.pop();
    const loaded = unpack(model.snapshot().encoded);
    const actual = loaded.forward(event.features, stop).a[stop + 1];
    pass.a[stop + 1].forEach((x, i) => close(x, actual[i], 1e-6));
    const blocked = new Set([model.edges[0].id]);
    assert.deepEqual(model.forward(event.features, stop, blocked).a, model.forward(event.features, stop, blocked, false, pass.inputDrive).a);
    const changed = event.features.map(x => x.slice());
    changed[stop].fill(50);
    assert.deepEqual(model.forward(event.features, stop - 1).a, model.forward(changed, stop - 1).a);
    if (encoder.endsWith('feedback')) {
      model.visualFeedback.fill(0);
      const gated = model.forward(event.features, stop).a;
      model.config = { ...model.config, encoder: 'hierarchical-motion' };
      assert.deepEqual(model.forward(event.features, stop).a, gated);
    }
  }
});

test('training state and audit restore reproduce subsequent updates and growth', () => {
  const { trainingState, auditState } = require('./synaptic_experiment');
  const settings = { ...config, encoder: 'hierarchical-feedback', mode: 'joint', interval: 4, confirmEvents: 2, candidates: 8 };
  const model = new SynapticNetwork(settings);
  for (let i = 0; i < 6; i++) model.train(makeEvent(7, 12, i, settings.encoder));
  const state = JSON.parse(JSON.stringify(trainingState(model)));
  const audit = JSON.parse(JSON.stringify(auditState(model)));
  const restored = SynapticNetwork.restore(state, audit);
  for (let i = 6; i < 10; i++) {
    const event = makeEvent(7, 12, i, settings.encoder);
    model.train(event); restored.train(event);
  }
  assert.deepEqual(restored.snapshot().encoded, model.snapshot().encoded);
  assert.deepEqual(auditState(restored), auditState(model));
  assert.deepEqual(restored.workHistory, model.workHistory);
});
