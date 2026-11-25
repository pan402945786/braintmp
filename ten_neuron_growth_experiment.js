'use strict';

const assert = require('node:assert/strict');

// 十个可能的位置。目标模式是亮、灭交替出现。
const positions = Array.from({ length: 10 }, (_, index) => index);
const data = positions.map((x) => [x, Number(x % 2 === 0)]);

// 一个神经元只负责一个局部位置。
// center 是它的感受野中心，输入正好落在中心时激活。
function activates(neuron, x) {
  return x === neuron.center;
}

// 十个神经元都看同一个输入。任意一个神经元激活，网络就输出亮。
function predict(neurons, x) {
  return Number(neurons.some((neuron) => activates(neuron, x)));
}

function outputs(neurons) {
  return positions.map((x) => predict(neurons, x));
}

function score(neurons) {
  return data.filter(([x, target]) => predict(neurons, x) === target).length;
}

// 尝试在尚未占用的十个位置长出一个神经元。
// 选择能带来最大正确率提升的位置，平分时取最靠左的位置。
function growOne(neurons) {
  let best = neurons;
  let bestScore = score(neurons);

  for (const center of positions) {
    if (neurons.some((neuron) => neuron.center === center)) continue;

    const candidate = [...neurons, { center }];
    const candidateScore = score(candidate);
    if (candidateScore > bestScore) {
      best = candidate;
      bestScore = candidateScore;
    }
  }

  return best;
}

function formatPattern(values) {
  return values.map((value) => (value ? '亮' : '灭')).join(' ');
}

console.log('十个位置的目标：');
console.log(`位置：   ${positions.join(' ')}`);
console.log(`目标：   ${formatPattern(data.map(([, target]) => target))}`);
console.log('');
console.log('结构：一个输入同时连接到所有候选神经元；神经元输出通过“任一个亮”读出。');
console.log('');

let growing = [];
const history = [{ count: growing.length, neurons: growing, score: score(growing) }];

while (growing.length < 10) {
  const next = growOne(growing);
  if (next.length === growing.length) break;
  growing = next;
  history.push({ count: growing.length, neurons: growing, score: score(growing) });
}

for (const entry of history) {
  const centers = entry.neurons.map((neuron) => neuron.center).join(',') || '无';
  console.log(`${entry.count} 个神经元：中心 [${centers}]，输出 ${formatPattern(outputs(entry.neurons))}，正确 ${entry.score}/10`);
}

const one = history.find((entry) => entry.count === 1);
const five = history.find((entry) => entry.count === 5);
assert.equal(one.score, 6);
assert.equal(five.score, 10);
assert.equal(five.neurons.length, 5);

// 固定容量组从一开始就拥有十个可用槽位，但学习后只启用五个需要的位置。
const fixedTen = data
  .filter(([, target]) => target === 1)
  .map(([center]) => ({ center }));
assert.equal(score(fixedTen), 10);

console.log('');
console.log(`生长结果：用 ${five.neurons.length} 个活动神经元达到 ${five.score}/10。`);
console.log(`固定容量 10：提前提供 10 个位置，但只启用中心 [${fixedTen.map((neuron) => neuron.center).join(',')}]，输出 ${formatPattern(outputs(fixedTen))}，正确 ${score(fixedTen)}/10。`);
