'use strict';

const assert = require('node:assert/strict');

// 两个端点亮，中间灭。最后才揭示右端点。
const samples = [[-1, 1], [0, 0], [1, 1]];
const firstTwo = samples.slice(0, 2);

// 每个神经元只检测一个方向。读出规则固定为“有一个激活就亮”。
function predict(neurons, x) {
  return Number(neurons.some(({ direction, threshold }) => direction * (x - threshold) >= 0));
}

function score(neurons, data) {
  return data.filter(([x, target]) => predict(neurons, x) === target).length;
}

// 对这些离散输入，把阈值放在观测点就能覆盖所有非空的单侧激活模式。
// 初始候选 [] 覆盖全灭，也让同分时优先保留原网络。
function bestAddition(neurons, data) {
  let best = neurons;
  for (const [threshold] of data) {
    for (const direction of [-1, 1]) {
      const candidate = [...neurons, { direction, threshold }];
      if (score(candidate, data) > score(best, data)) best = candidate;
    }
  }
  return best;
}

const initial = bestAddition([], firstTwo);
assert.equal(initial.length, 1);
assert.equal(score(initial, firstTwo), 2);
console.log('只看左、中两点：1 个神经元，答对 2/2。');
console.log('加入右点后，目标输出为 1 0 1。');

const fixed = bestAddition([], samples); // 单神经元可在全部三点上重新选方向和阈值。
const grown = score(initial, samples) < samples.length ? bestAddition(initial, samples) : initial;
assert.equal(score(fixed, samples), 2);
assert.equal(grown.length, 2);
assert.equal(score(grown, samples), 3);
for (const [label, neurons] of [['固定 1 个，重新拟合', fixed], ['保留原神经元，再长 1 个', grown]]) {
  const outputs = samples.map(([x]) => predict(neurons, x));
  console.log(`${label}：输出 ${outputs.join(' ')}，答对 ${score(neurons, samples)}/3。`);
}
