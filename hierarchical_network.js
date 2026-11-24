'use strict';

// One shared 3x3 motion kernel acts on the 8x8 frame-difference map.
// Feedback uses a previous high-level state and gates each spatial channel.
const N = 32, C = 25, LEAK = 0.45, CHANNELS = 7, SIZE = 64;
const neighbor = (pos, k) => {
  const y = Math.max(0, Math.min(7, Math.floor(pos / 8) + Math.floor(k / 3) - 1));
  const x = Math.max(0, Math.min(7, pos % 8 + k % 3 - 1));
  return y * 8 + x;
};
function middle(frame, kernel) {
  const values = new Float64Array(449);
  values.set(frame.subarray(0, 384));
  values[448] = frame[384];
  for (let p = 0; p < SIZE; p++) {
    let z = 0;
    for (let k = 0; k < 9; k++) z += kernel[k] * frame[320 + neighbor(p, k)];
    values[384 + p] = Math.tanh(z);
  }
  return values;
}
function project(features, stop) {
  return features.slice(0, stop + 1).map(frame => {
    const values = middle(frame, this.motionKernel);
    const channels = new Float64Array(CHANNELS * N);
    const audio = new Float64Array(N);
    for (let j = 0; j < N; j++) {
      const offset = j * this.inputs;
      audio[j] = this.input[offset + 448] * values[448];
      for (let c = 0; c < CHANNELS; c++) {
        let sum = 0;
        for (let p = 0; p < SIZE; p++) sum += this.input[offset + c * SIZE + p] * values[c * SIZE + p];
        channels[c * N + j] = sum;
      }
    }
    return { values, channels, audio };
  });
}
function forward(features, stop, blocked, countWork = false, inputDrive) {
  const edges = this.edges.filter(e => e.deleted === null && !blocked?.has(e.id));
  const a = [new Float64Array(N)], activation = [], gates = [], projection = inputDrive ?? this.project(features, stop);
  for (let t = 0; t <= stop; t++) {
    const gate = new Float64Array(CHANNELS).fill(1);
    if (this.config.encoder === 'hierarchical-feedback' && t > 0) {
      for (let c = 0; c < CHANNELS; c++) {
        let z = 0;
        for (let i = 0; i < N; i++) z += this.visualFeedback[c * N + i] * a[t][i];
        gate[c] += 0.25 * Math.tanh(z);
      }
    }
    const sums = projection[t].audio.slice();
    for (let c = 0; c < CHANNELS; c++) for (let j = 0; j < N; j++) sums[j] += gate[c] * projection[t].channels[c * N + j];
    for (const e of edges) sums[e.j] += e.w * a[t][e.i];
    const nonlinearity = sums.map((sum, j) => Math.tanh(this.gain[j] * sum));
    activation.push(nonlinearity); gates.push(gate);
    a.push(nonlinearity.map((value, j) => LEAK * a[t][j] + (1 - LEAK) * value));
  }
  if (countWork) this.work += (stop + 1) * ((inputDrive ? 0 : N * this.inputs + SIZE * 9) + CHANNELS * N * 2 + edges.length + N)
    + (this.config.encoder === 'hierarchical-feedback' ? stop * CHANNELS * N : 0);
  return { a, activation, edges, gates, inputDrive: projection };
}
function gradient(features, prefixes, target, precomputed, countWork = false) {
  const stop = prefixes.at(-1);
  const { a, activation, edges, gates, inputDrive } = precomputed ?? this.forward(features, stop, null, countWork);
  const inGrad = new Float64Array(this.input.length), outGrad = new Float64Array(this.output.length), biasGrad = new Float64Array(C);
  const edgeGrad = new Float64Array(edges.length), motionGrad = new Float64Array(9), visualFeedbackGrad = new Float64Array(CHANNELS * N);
  const delta = Array.from({ length: stop + 1 }, () => new Float64Array(N));
  const b = Array.from({ length: stop + 1 }, () => new Float64Array(N));
  let eventLoss = 0;
  for (const t of prefixes) {
    const prob = this.read(a[t + 1]);
    eventLoss -= Math.log(Math.max(prob[target], Number.MIN_VALUE)) / prefixes.length;
    for (let c = 0; c < C; c++) {
      const g = (prob[c] - Number(c === target)) / prefixes.length;
      biasGrad[c] += g;
      for (let j = 0; j < N; j++) { outGrad[c * N + j] += g * a[t + 1][j]; delta[t][j] += g * this.output[c * N + j]; }
    }
  }
  for (let t = stop; t >= 0; t--) {
    const values = inputDrive[t].values, gate = gates[t];
    const motionSignals = new Float64Array(SIZE), gateSignals = new Float64Array(CHANNELS);
    for (let j = 0; j < N; j++) {
      const signal = delta[t][j] * (1 - LEAK) * this.gain[j] * (1 - activation[t][j] ** 2);
      b[t][j] = signal;
      if (t > 0) delta[t - 1][j] += LEAK * delta[t][j];
      const offset = j * this.inputs;
      inGrad[offset + 448] += signal * values[448];
      for (let c = 0; c < CHANNELS; c++) {
        gateSignals[c] += signal * inputDrive[t].channels[c * N + j];
        for (let p = 0; p < SIZE; p++) {
          const k = c * SIZE + p;
          inGrad[offset + k] += signal * gate[c] * values[k];
          if (c === 6) motionSignals[p] += signal * gate[c] * this.input[offset + k];
        }
      }
    }
    for (let p = 0; p < SIZE; p++) for (let k = 0; k < 9; k++) {
      motionGrad[k] += motionSignals[p] * (1 - values[384 + p] ** 2) * features[t][320 + neighbor(p, k)];
    }
    if (t > 0 && this.config.encoder === 'hierarchical-feedback') {
      for (let c = 0; c < CHANNELS; c++) {
        const tanh = (gate[c] - 1) / 0.25;
        const signal = gateSignals[c] * 0.25 * (1 - tanh * tanh);
        for (let i = 0; i < N; i++) {
          visualFeedbackGrad[c * N + i] += signal * a[t][i];
          delta[t - 1][i] += signal * this.visualFeedback[c * N + i];
        }
      }
    }
    for (let k = 0; k < edges.length; k++) {
      const e = edges[k]; edgeGrad[k] += b[t][e.j] * a[t][e.i];
      if (t > 0) delta[t - 1][e.i] += e.w * b[t][e.j];
    }
  }
  if (countWork) this.work += (stop + 1) * (N * this.inputs + SIZE * 9 * 2 + CHANNELS * N * 4 + 2 * edges.length + 2 * N)
    + (this.config.encoder === 'hierarchical-feedback' ? stop * CHANNELS * N * 2 : 0) + prefixes.length * 3 * C * N;
  return { inGrad, outGrad, biasGrad, edgeGrad, motionGrad, visualFeedbackGrad, edges, a: a.slice(1), b, eventLoss };
}
module.exports = { project, forward, gradient };
