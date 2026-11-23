# Predictive growth experiment

`predictive_growth_experiment.js` is a dependency free implementation of the
first round described in `predictive_growth_experiment.md`.

It contains:

- a seeded 48-frame falling-ball generator with 32×32 grayscale frames and a
  synchronized impact sound;
- a sparse recurrent predictor with edge birth time, stability, contribution,
  and deletion metadata;
- online post-event learning from saved prefixes;
- fixed-initial, random-growth, experience-growth, fixed-capacity, and time
  prior controls;
- shared test events and shared random observation cutoffs;
- NLL in bits, median frame error, anticipation curves, mature-edge blocking,
  model size, and estimated compression break-even point.

Run a quick smoke experiment:

```powershell
node .\predictive_growth_experiment.js `
  --train-events 512 `
  --dev-events 128 `
  --test-events 256 `
  --checkpoints 0,128,512 `
  --output .\predictive_growth_results.json
```

For the formal run, increase the counts to the values registered in the
experiment document, for example `--train-events 8192 --test-events 2000`.
The implementation keeps the same seeds and per-event cutoffs across methods,
so method differences are paired comparisons.

The module can also be imported:

```js
const { SceneGenerator, SparseGrowingRNN, runExperiment } = require('./predictive_growth_experiment');
const report = runExperiment({ trainEvents: 128, testEvents: 64 });
```

For the registered multi-seed comparison, use `runStudy({ seeds: 10, ... })`.
It returns each seed-level report plus paired bootstrap 95% intervals for the
losses and for the random-growth versus experience-growth difference.

The defaults are deliberately smoke-test sized. They make the implementation
easy to validate on a plain Node installation; the reported metrics should be
re-run at the registered scale before drawing scientific conclusions.

