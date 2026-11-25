'use strict';

const assert = require('node:assert/strict');

const INPUTS = 5;
const OUTPUTS = 3;
const MAX_NEURONS = 10;
const SEQUENCE_LENGTH = 12;
const CONTEXTS = 5;
const ROUTES = [
  [0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2],
  [0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1],
  [1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0],
  [2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0],
  [1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
];

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[best]) best = i;
  }
  return best;
}

function zeros(length) {
  return Array.from({ length }, () => 0);
}

class RNG {
  constructor(seed = 1) {
    this.state = (seed >>> 0) || 1;
  }

  next() {
    let x = this.state;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.state = x >>> 0;
    return this.state / 0x100000000;
  }

  uniform(min = 0, max = 1) {
    return min + (max - min) * this.next();
  }

  int(min, max) {
    return Math.floor(this.uniform(min, max + 1));
  }
}

function randomMatrix(rows, columns, rng, scale) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => rng.uniform(-scale, scale))
  );
}

function makeEpisode(rng) {
  const context = rng.int(0, CONTEXTS - 1);
  const inputs = [];
  const targets = ROUTES[context].slice();

  for (let t = 0; t < SEQUENCE_LENGTH; t++) {
    const input = zeros(INPUTS);
    if (t === 0) {
      // A brief colored landmark identifies the route. The small noise models
      // imperfect sensing. All later observations are the same blank corridor.
      input[context] = 1;
      for (let i = 0; i < INPUTS; i++) input[i] += rng.uniform(-0.04, 0.04);
    } else {
      for (let i = 0; i < INPUTS; i++) input[i] = rng.uniform(-0.04, 0.04);
    }
    inputs.push(input);
  }

  return { context, inputs, targets };
}

function makeDataset(seed, episodes) {
  const rng = new RNG(seed);
  return Array.from({ length: episodes }, () => makeEpisode(rng));
}

class RecurrentPredictor {
  constructor(activeNeurons, seed = 1) {
    const rng = new RNG(seed);
    this.activeNeurons = activeNeurons;
    // Draw in the same order for both widths so that they share initial weights.
    this.inputWeights = randomMatrix(MAX_NEURONS, INPUTS, rng, 0.35).slice(0, activeNeurons);
    this.recurrentWeights = randomMatrix(MAX_NEURONS, MAX_NEURONS, rng, 0.12)
      .slice(0, activeNeurons).map((row) => row.slice(0, activeNeurons));
    this.hiddenBias = zeros(activeNeurons);
    this.outputWeights = randomMatrix(OUTPUTS, MAX_NEURONS, rng, 0.25)
      .map((row) => row.slice(0, activeNeurons));
    this.outputBias = zeros(OUTPUTS);
  }

  grow(seed) {
    assert(this.activeNeurons < MAX_NEURONS);
    const rng = new RNG(seed);
    this.inputWeights.push(Array.from({ length: INPUTS }, () => rng.uniform(-0.08, 0.08)));
    for (const row of this.recurrentWeights) row.push(0);
    this.recurrentWeights.push(zeros(this.activeNeurons + 1));
    for (const row of this.outputWeights) row.push(rng.uniform(-0.02, 0.02));
    this.hiddenBias.push(0);
    this.activeNeurons++;
  }

  forward(episode, resetMemory = false) {
    const hidden = [];
    const probabilities = [];
    let previous = zeros(this.activeNeurons);

    for (const input of episode.inputs) {
      const current = zeros(this.activeNeurons);
      for (let i = 0; i < this.activeNeurons; i++) {
        let value = this.hiddenBias[i];
        for (let j = 0; j < INPUTS; j++) value += this.inputWeights[i][j] * input[j];
        for (let j = 0; j < this.activeNeurons; j++) value += this.recurrentWeights[i][j] * previous[j];
        current[i] = Math.tanh(value);
      }

      const logits = zeros(OUTPUTS);
      for (let k = 0; k < OUTPUTS; k++) {
        logits[k] = this.outputBias[k];
        for (let i = 0; i < this.activeNeurons; i++) logits[k] += this.outputWeights[k][i] * current[i];
      }
      const largest = Math.max(...logits);
      const exponentials = logits.map((logit) => Math.exp(logit - largest));
      const total = exponentials.reduce((sum, value) => sum + value, 0);
      probabilities.push(exponentials.map((value) => value / total));
      hidden.push(current);
      if (!resetMemory) previous = current;
    }

    return { hidden, probabilities };
  }

  trainEpisode(episode, learningRate = 0.015) {
    const { hidden, probabilities } = this.forward(episode);
    const inputGradient = Array.from({ length: this.activeNeurons }, () => zeros(INPUTS));
    const recurrentGradient = Array.from({ length: this.activeNeurons }, () => zeros(this.activeNeurons));
    const hiddenBiasGradient = zeros(this.activeNeurons);
    const outputGradient = Array.from({ length: OUTPUTS }, () => zeros(this.activeNeurons));
    const outputBiasGradient = zeros(OUTPUTS);
    let loss = 0;
    let nextHiddenGradient = zeros(this.activeNeurons);

    for (let t = SEQUENCE_LENGTH - 1; t >= 0; t--) {
      const target = episode.targets[t];
      const probability = Math.max(probabilities[t][target], 1e-12);
      loss -= Math.log(probability);

      const logitsGradient = probabilities[t].slice();
      logitsGradient[target] -= 1;
      const hiddenGradient = nextHiddenGradient.slice();
      for (let k = 0; k < OUTPUTS; k++) {
        outputBiasGradient[k] += logitsGradient[k];
        for (let i = 0; i < this.activeNeurons; i++) {
          outputGradient[k][i] += logitsGradient[k] * hidden[t][i];
          hiddenGradient[i] += logitsGradient[k] * this.outputWeights[k][i];
        }
      }

      const recurrentSource = t === 0 ? zeros(this.activeNeurons) : hidden[t - 1];
      const derivative = hidden[t].map((value) => 1 - value * value);
      const previousGradient = zeros(this.activeNeurons);
      for (let i = 0; i < this.activeNeurons; i++) {
        hiddenGradient[i] *= derivative[i];
        for (let j = 0; j < this.activeNeurons; j++) {
          recurrentGradient[i][j] += hiddenGradient[i] * recurrentSource[j];
          if (t > 0) previousGradient[j] += hiddenGradient[i] * this.recurrentWeights[i][j];
        }
        hiddenBiasGradient[i] += hiddenGradient[i];
        for (let j = 0; j < INPUTS; j++) inputGradient[i][j] += hiddenGradient[i] * episode.inputs[t][j];
      }
      nextHiddenGradient = previousGradient;
    }

    // Apply the update only after differentiating the whole sequence.
    for (let i = 0; i < this.activeNeurons; i++) {
      this.hiddenBias[i] -= learningRate * hiddenBiasGradient[i];
      for (let j = 0; j < INPUTS; j++) this.inputWeights[i][j] -= learningRate * inputGradient[i][j];
      for (let j = 0; j < this.activeNeurons; j++) this.recurrentWeights[i][j] -= learningRate * recurrentGradient[i][j];
    }
    for (let k = 0; k < OUTPUTS; k++) {
      this.outputBias[k] -= learningRate * outputBiasGradient[k];
      for (let i = 0; i < this.activeNeurons; i++) this.outputWeights[k][i] -= learningRate * outputGradient[k][i];
    }

    return loss / SEQUENCE_LENGTH;
  }

  evaluate(dataset, resetMemory = false) {
    let totalLoss = 0;
    let correct = 0;
    let count = 0;
    let delayedCorrect = 0;
    let delayedCount = 0;
    for (const episode of dataset) {
      const result = this.forward(episode, resetMemory);
      for (let t = 0; t < SEQUENCE_LENGTH; t++) {
        totalLoss -= Math.log(Math.max(result.probabilities[t][episode.targets[t]], 1e-12));
        correct += Number(argmax(result.probabilities[t]) === episode.targets[t]);
        count++;
        if (t > 0) {
          delayedCorrect += Number(argmax(result.probabilities[t]) === episode.targets[t]);
          delayedCount++;
        }
      }
    }
    return {
      loss: totalLoss / count,
      accuracy: correct / count,
      delayedAccuracy: delayedCorrect / delayedCount,
    };
  }
}

function train({ neurons, seed, trainSet, validationSet, allowGrowth = false, epochs = 180 }) {
  const model = new RecurrentPredictor(neurons, seed);
  const history = [];
  const growthEvents = [];
  let bestLoss = Infinity;
  let staleEpochs = 0;

  for (let epoch = 1; epoch <= epochs; epoch++) {
    for (const episode of trainSet) model.trainEpisode(episode);
    const metrics = model.evaluate(validationSet);
    history.push({ epoch, neurons: model.activeNeurons, ...metrics });

    if (!allowGrowth) continue;
    if (metrics.loss < bestLoss - 0.003) {
      bestLoss = metrics.loss;
      staleEpochs = 0;
    } else {
      staleEpochs++;
    }

    if (staleEpochs >= 8 && model.activeNeurons < MAX_NEURONS && epoch < epochs) {
      model.grow(seed + epoch * 7919);
      growthEvents.push({ epoch, neurons: model.activeNeurons, validationLoss: metrics.loss });
      staleEpochs = 0;
    }
  }
  return { model, history, growthEvents };
}

function summarize(label, { model, growthEvents }, testSet) {
  const metrics = model.evaluate(testSet);
  return { label, finalNeurons: model.activeNeurons, ...metrics, growthEvents };
}

function runExperiment(seed = 7) {
  const trainSet = makeDataset(20261125, 60);
  const validationSet = makeDataset(70261125, 80);
  const testSet = makeDataset(90261125, 80);
  const shared = { seed, trainSet, validationSet };
  const fixed2 = train({ ...shared, neurons: 2 });
  const fixed10 = train({ ...shared, neurons: 10 });
  const growing = train({ ...shared, neurons: 2, allowGrowth: true });

  const report = {
    task: 'route-context-next-event-prediction',
    seed,
    architecture: {
      inputChannels: INPUTS,
      maxNeurons: MAX_NEURONS,
      outputClasses: ['无事件', '左侧地标', '右侧地标'],
      recurrent: true,
    },
    controls: [
      summarize('fixed-2', fixed2, testSet),
      summarize('growing-2-to-10', growing, testSet),
      summarize('fixed-10', fixed10, testSet),
    ],
    growingWithMemoryReset: growing.model.evaluate(testSet, true),
    validationSnapshots: growing.history.filter((entry) => [1, 20, 40, 60, 90, 120, 150, 180].includes(entry.epoch)),
  };

  return report;
}

if (require.main === module) {
  const report = process.argv.includes('--study')
    ? [7, 8, 9, 10, 11].map(runExperiment)
    : runExperiment();
  console.log(JSON.stringify(report, null, 2));
}

module.exports = { RecurrentPredictor, makeDataset, runExperiment };
