'use strict';

const fs = require('node:fs');
const path = require('node:path');

const style = {
  uniform: ['均匀先验', '#8091a5'], prior: ['频数先验', '#606d7d'],
  fixed: ['固定结构', '#123c72'], random: ['随机选边', '#b54819'],
  hebb: ['Hebb 选边', '#6a4092'], bidirectional: ['两端选边', '#16748a'],
  joint: ['联合选边', '#147456'], shuffled: ['错配反馈选边', '#92435e'], capacity: ['固定容量', '#5c5a21'],
};

function loadPlotData(dir) {
  const summary = JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8'));
  const seeds = summary.seeds.map(({ seed }) => {
    const seedDir = path.join(dir, `seed-${seed}`);
    const report = JSON.parse(fs.readFileSync(path.join(seedDir, 'report.json'), 'utf8'));
    const checkpoints = Object.entries(report.checkpoints).map(([event, point]) => ({ event: Number(event),
      methods: Object.fromEntries(Object.entries(point.methods).map(([method, row]) => [method, {
        bits: row.scores[0].bits, anticipation: row.scores[0].anticipation, active: row.activeEdges, mature: row.mature,
      }])), prior: { bits: point.prior[0].bits, anticipation: point.prior[0].anticipation } }));
    const audits = {};
    for (const filename of fs.readdirSync(seedDir).filter(file => file.endsWith('-edges.json'))) {
      const audit = JSON.parse(fs.readFileSync(path.join(seedDir, filename), 'utf8'));
      const method = filename.slice(0, -11);
      audits[method] = {
        edges: audit.edges.filter(edge => edge.born > 0).map(({ id, i, j, born, deleted, batch }) => ({ id, i, j, born, deleted, batch })),
        batches: audit.batches.map(({ id, born, matured, windows }) => ({ id, born, matured, windows })),
        proposals: audit.proposals.map(({ event, skipped, candidates }) => ({ event, skipped,
          selected: candidates?.filter(candidate => candidate.selected).map(({ i, j, q, hebb, joint, selected }) => ({ i, j, q, hebb, joint, selected })) })),
      };
    }
    return { seed, uniform: { bits: report.uniform.bits, anticipation: report.uniform.anticipation }, checkpoints, audits };
  });
  const methods = Object.keys(seeds[0].checkpoints[0].methods);
  const series = ['uniform', 'prior', ...methods].map((method, i) => ({ method,
    label: style[method]?.[0] ?? method, color: style[method]?.[1] ?? ['#3868b1', '#665489', '#92682f'][i % 3] }));
  return { seeds, series, config: { profile: summary.config.profile, interval: summary.config.interval, train: summary.config.train } };
}

// Kept as a self-contained browser function. The generated report can be opened offline.
function renderReport(data) {
  const id = name => document.getElementById(name);
  const svgElement = (name, options, parent) => {
    const el = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(options)) el.setAttribute(key, value);
    parent.append(el);
    return el;
  };
  const text = (parent, content, x, y, options = {}) => {
    const label = svgElement('text', { x, y, ...options }, parent);
    label.textContent = content;
  };
  const mean = values => values.reduce((a, b) => a + b, 0) / values.length;
  const legend = id('legend');
  data.series.forEach(({ label, color }) => {
    const item = document.createElement('span');
    item.style.setProperty('--series', color);
    item.textContent = label;
    legend.append(item);
  });

  function chart(container, lines, { min, max, xLabel, yLabel, ticks, percent }) {
    const width = 750, height = 300, left = 65, right = width - 25, top = 28, bottom = height - 49;
    const xs = lines[0].points.map(point => point[0]);
    const lowX = Math.min(...xs), highX = Math.max(...xs);
    const px = x => left + (right - left) * (x - lowX) / Math.max(1, highX - lowX);
    const py = y => bottom - (bottom - top) * (y - min) / (max - min);
    const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${xLabel}与${yLabel}` }, container);
    for (let k = 0; k <= 4; k++) {
      const y = min + k * (max - min) / 4;
      svgElement('line', { x1: left, x2: right, y1: py(y), y2: py(y), stroke: '#dce4ed' }, svg);
      text(svg, percent ? `${Math.round(100 * y)}%` : y.toFixed(2), left - 10, py(y) + 4, { 'text-anchor': 'end' });
    }
    for (const x of ticks) text(svg, x, px(x), bottom + 20, { 'text-anchor': 'middle' });
    text(svg, yLabel, left, 16);
    text(svg, xLabel, (left + right) / 2, height - 6, { 'text-anchor': 'middle' });
    for (const { points, color, method } of lines) {
      const line = svgElement('polyline', { fill: 'none', stroke: color, 'stroke-width': method === 'joint' ? 3.5 : 2,
        points: points.map(([x, y]) => `${px(x)},${py(y)}`).join(' ') }, svg);
      if (method === 'prior' || method === 'uniform') line.setAttribute('stroke-dasharray', '5 5');
      points.forEach(([x, y]) => svgElement('circle', { cx: px(x), cy: py(y), r: 3.5, fill: color }, svg));
    }
  }

  const steps = data.seeds[0].checkpoints.map(row => row.event);
  const get = (seed, row, method) => method === 'uniform' ? seed.uniform : method === 'prior' ? row.prior : row.methods[method];
  const bits = data.series.map(series => ({ ...series, points: steps.map((step, k) => [step,
    mean(data.seeds.map(seed => get(seed, seed.checkpoints[k], series.method).bits))]) }));
  const y = bits.flatMap(series => series.points.map(point => point[1]));
  const stepTicks = steps.filter((step, k) => k === 0 || k === steps.length - 1 ||
    Math.min(step - steps[0], steps.at(-1) - step) > steps.at(-1) * 0.04 && (steps.length <= 8 || k % Math.ceil(steps.length / 6) === 0));
  chart(id('loss'), bits, { min: Math.floor(Math.min(...y) * 4) / 4 - 0.05,
    max: Math.ceil(Math.max(...y) * 4) / 4 + 0.05, xLabel: '训练事件数', yLabel: '预测损失（比特／事件）', ticks: stepTicks });
  const horizons = [2, 4, 6, 8, 10, 12];
  chart(id('horizons'), data.series.map(series => ({ ...series, points: horizons.map(t => [t,
    mean(data.seeds.map(seed => get(seed, seed.checkpoints.at(-1), series.method).anticipation[t]))]) })),
  { min: 0, max: 1, xLabel: '距离撞击的帧数', yLabel: '误差在 ±1 帧以内的比例', ticks: horizons, percent: true });

  const growSeeds = data.seeds.filter(seed => Object.keys(seed.audits).length);
  if (!growSeeds.length) { id('growth-panel').hidden = true; return; }
  const seedSelect = id('seed'), methodSelect = id('method'), round = id('round'), grid = id('grid');
  function addOption(select, value, label) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    select.append(option);
  }
  growSeeds.forEach(seed => addOption(seedSelect, seed.seed, seed.seed));
  const getSeed = () => growSeeds.find(seed => seed.seed === Number(seedSelect.value));
  function selectSeed() {
    methodSelect.replaceChildren();
    const list = Object.keys(getSeed().audits);
    for (const method of list) addOption(methodSelect, method, data.series.find(series => series.method === method).label);
    methodSelect.value = list.includes('joint') ? 'joint' : list[0];
    configureRound();
  }
  const getAudit = () => getSeed().audits[methodSelect.value];
  function configureRound() {
    const proposals = getAudit().proposals;
    round.max = Math.max(0, proposals.length - 1);
    round.value = 0;
    round.disabled = proposals.length === 0;
    renderRound();
  }
  function renderRound() {
    const audit = getAudit();
    const proposal = audit.proposals[Number(round.value)];
    grid.replaceChildren();
    id('selection').replaceChildren();
    if (!proposal) { id('round-label').textContent = '当前运行尚未达到第一次选边'; return; }
    const event = proposal.event;
    id('round-label').textContent = `${event} 个训练事件后`;
    const batches = new Map(audit.batches.map(batch => [batch.id, batch]));
    const colors = { trial: '#3d76a7', mature: '#147456', selected: '#c65d16', deleted: '#7b8390' };
    const active = audit.edges.filter(edge => edge.born <= event && (edge.deleted === null || edge.deleted > event));
    const mature = active.filter(edge => batches.get(edge.batch).matured !== null && batches.get(edge.batch).matured <= event).length;
    id('counts').textContent = `新增边中，${active.length} 条在用，其中 ${mature} 条已经成熟`;
    const svg = svgElement('svg', { viewBox: '0 0 610 575', role: 'img', 'aria-label': '32 个隐藏单元之间的新增连接矩阵' }, grid);
    const x = i => 68 + i * 15, y = j => 46 + j * 15;
    svgElement('rect', { x: x(0), y: y(0), width: 480, height: 480, fill: '#f4f7fb', stroke: '#cbd6e2' }, svg);
    for (let i = 0; i < 32; i += 4) {
      text(svg, i, x(i) + 7.5, 37, { 'text-anchor': 'middle' });
      text(svg, i, 59, y(i) + 12, { 'text-anchor': 'end' });
      if (!i) continue;
      svgElement('line', { x1: x(i), y1: y(0), x2: x(i), y2: y(32), stroke: '#dde6f0' }, svg);
      svgElement('line', { x1: x(0), y1: y(i), x2: x(32), y2: y(i), stroke: '#dde6f0' }, svg);
    }
    text(svg, '源单元', 310, 16, { 'text-anchor': 'middle' });
    text(svg, '目标单元', 18, 273, { transform: 'rotate(-90 18 273)', 'text-anchor': 'middle' });
    const positions = new Map();
    for (const edge of audit.edges) if (edge.born <= event) positions.set(`${edge.i}:${edge.j}`, edge);
    for (const edge of positions.values()) {
      const batch = batches.get(edge.batch);
      const state = edge.deleted !== null && edge.deleted <= event ? 'deleted' : edge.born === event ? 'selected'
        : batch.matured !== null && batch.matured <= event ? 'mature' : 'trial';
      const mark = svgElement('rect', { x: x(edge.i) + 2, y: y(edge.j) + 2, width: 11, height: 11,
        fill: state === 'deleted' ? 'none' : colors[state], stroke: colors[state], rx: 2 }, svg);
      svgElement('title', {}, mark).textContent = `H${edge.i} → H${edge.j}，出生于 ${edge.born} 个事件后`;
    }
    for (const [i, [key, label]] of Object.entries([['selected', '本轮试接'], ['trial', '观察中'], ['mature', '已成熟'], ['deleted', '已撤回']])) {
      const tx = 68 + Number(i) * 119;
      svgElement('rect', { x: tx, y: 548, width: 11, height: 11, stroke: colors[key],
        fill: key === 'deleted' ? 'none' : colors[key], rx: 2 }, svg);
      text(svg, label, tx + 17, 559);
    }
    const list = id('selection');
    if (proposal.skipped) list.textContent = proposal.skipped === 'capacity' ? '达到总边数上限' : '没有符合入度约束的候选';
    for (const candidate of proposal.selected ?? []) {
      const row = document.createElement('li');
      row.textContent = `H${candidate.i} → H${candidate.j} · ${candidate.selected === 'explore' ? '随机探索' : '按组别规则选入'} · Hebb ${candidate.hebb.toFixed(3)} · 反馈 ${candidate.q.toExponential(2)}`;
      list.append(row);
    }
    const thisBatch = audit.batches.find(batch => batch.born === event);
    id('gains').textContent = thisBatch?.windows.length
      ? `当前试接批次的后续窗口收益（比特／事件） ${thisBatch.windows.map(window => window.gainBits.toFixed(4)).join('，')}`
      : '该批次尚无后续窗口收益记录';
  }
  seedSelect.addEventListener('change', selectSeed);
  methodSelect.addEventListener('change', configureRound);
  round.addEventListener('input', renderRound);
  selectSeed();
}

function buildReport(dir) {
  const data = loadPlotData(dir);
  const payload = JSON.stringify(data).replace(/</g, '\\u003c');
  const page = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>突触生长实验</title><style>
:root{font-family: "Segoe UI", "Microsoft YaHei", sans-serif;color:#14263a;background:#f3f6fb}*{box-sizing:border-box}
body{max-width:1040px;margin:auto;padding:28px 18px}h1{margin:0 0 8px;font-size:32px}h2{margin:0 0 16px;font-size:20px}p{line-height:1.65}
section{background:#fff;border:1px solid #dbe3ee;border-radius:12px;margin:18px 0;padding:18px}svg{width:100%;max-height:620px;display:block}
svg text{font-family:inherit;font-size:15px;fill:#253c57}#legend,.controls{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center}
#legend span{display:inline-flex;align-items:center;gap:6px}#legend span::before{content:'';width:14px;height:3px;background:var(--series)}
.controls label{display:flex;gap:6px;align-items:center}select{font:inherit;padding:5px}input[type=range]{width:min(100%,560px);display:block;margin:12px 0}li{margin:6px 0}
</style></head><body><main><h1>落球预测与突触生长</h1><p>${data.config.profile === 'formal' ? '正式运行' : '探索运行'} · ${data.seeds.length} 个种子 · 曲线为种子均值</p>
<div id="legend" aria-label="曲线图例"></div>
<section><h2>预测损失随训练变化</h2><div id="loss"></div></section>
<section><h2>最终检查点的提前预测</h2><div id="horizons"></div></section>
<section id="growth-panel"><h2>新增连接回放</h2><div class="controls"><label>种子 <select id="seed"></select></label><label>方法 <select id="method"></select></label></div>
<label for="round" id="round-label">生长轮次</label><input id="round" type="range" min="0" max="0" value="0"><p id="counts" aria-live="polite"></p>
<div id="grid"></div><ul id="selection"></ul><p id="gains"></p></section></main>
<script>(${renderReport.toString()})(${payload});</script></body></html>`;
  const filename = path.join(dir, 'report.html');
  fs.writeFileSync(filename, page);
  return filename;
}

if (require.main === module) {
  if (!process.argv[2]) { console.error('Usage: node plot_synaptic_experiment.js <run-directory>'); process.exitCode = 1; }
  else console.log(buildReport(path.resolve(process.argv[2])));
}

module.exports = { buildReport };
