'use strict';

const fs = require('node:fs');
const path = require('node:path');
const [comparisonFile, screenFile, outputFile] = process.argv.slice(2);
if (!comparisonFile || !screenFile || !outputFile) throw new Error('Usage: node write_attention_report.js COMPARISON_JSON SCREEN_JSON OUTPUT_MD');
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const result = read(comparisonFile), screen = read(screenFile);
if (result.codeHash !== screen.codeHash) throw new Error('Screen and long-run sources differ');
const format = x => x.toFixed(4);
const mean = xs => xs.reduce((sum, x) => sum + x, 0) / xs.length;
const main = result.comparisons.vsMotion, secondary = result.comparisons.vsFeedback;
const last = result.learningCurves.at(-1);
const bothPass = main.developmentCriterion.passed && secondary.developmentCriterion.passed;
const conclusion = bothPass ? '本轮达到两个开发比较的预设标准，值得用新种子和独立正式测试继续验证。' :
  main.developmentCriterion.passed ? '相对运动模型达到预设开发标准，但相对原反馈模型尚未达到；不能把收益明确归因于新增的归一化。' :
  '本轮未达到相对运动模型的预设开发标准，当前证据不足以用该版本替代现有运动方案。';
const seedRows = last.perSeed.map(({seed,bits}, i) => `| ${seed} | ${format(bits.motion)} | ${format(bits.feedback)} | ${format(bits.attention)} | ${format(main.sameEvents.perSeed[i])} | ${format(secondary.sameEvents.perSeed[i])} |`).join('\n');
const curveRows = result.learningCurves.map(({events,meanBits:b}) => `| ${events} | ${format(b.motion)} | ${format(b.feedback)} | ${format(b.attention)} | ${format(b.motion - b.attention)} | ${format(b.feedback - b.attention)} |`).join('\n');
const comparisonText = (label, row) => `${label}：平均改善 **${format(row.sameEvents.mean)} bit/事件**；${row.developmentCriterion.positiveSeeds}/${row.rows.length} 个种子改善；配对 bootstrap 95% 区间 [${row.sameEvents.bounds.map(format).join(', ')}]；预设标准${row.developmentCriterion.passed ? '达到' : '未达到'}。`;
const diagnosticRows = result.diagnostics.map(({seed,attention:a}) => `| ${seed} | ${format(a.channels.brightness.meanGain)} | ${format(a.channels.frameDifference.meanGain)} | ${format(a.channels.learnedMotion.meanGain)} | ${format(a.disabledMinusEnabledBits)} |`).join('\n');
const relativeComparison = path.relative(path.dirname(outputFile), comparisonFile).replaceAll('\\', '/');
const text = `# 注意力优化开发结果（2026-11-24）

${conclusion}

## 做了什么

在现有可学习运动模型上，新增 hierarchical-attention：用上一帧高层状态产生七个视觉特征通道的正增益，再除以七通道的平均值。每帧平均增益为 1，让不同特征形成相对竞争；首帧中性、声音不受门控。这是特征通道注意力，没有实现空间选点。

原反馈模型允许各通道独立增强或减弱；新版本复用同一套 224 个反馈权重，增加 0 个参数。新旧反馈组的输入、初始化、可训练参数数组大小、部署尺寸相同。相对运动模型则多出有效反馈参数，因此运动比较本身不能排除模型容量的贡献。

设计参考 Reynolds 与 Heeger（2009），*The Normalization Model of Attention*，DOI 10.1016/j.neuron.2009.01.002；仅借鉴归一化的思路。具体设计及预设标准见 [实验设计](attention_design_2026-11-24.md)。

## 实验设置

三个模型均为固定结构，每种子 ${result.config.train.toLocaleString('en-US')} 个训练事件、${result.config.samples} 个开发事件，共 ${result.config.seeds} 个配对训练种子。学习率 ${result.config.rate}，沿用原任务、开发场景和 BPTT。每组从头训练；本轮全部配对运行源码哈希相同：${result.codeHash}。

主指标是最终检查点 NLL（bit/事件，越低越好）；差值为对照减注意力，正值有利于注意力。运行前约定平均改善至少 0.05 且全部种子方向为正，作为进一步验证的开发标准；它不是统计显著性或正式测试标准。

## 最终结果

| 随机种子 | 运动 NLL | 原反馈 NLL | 注意力 NLL | 运动减注意力 | 原反馈减注意力 |
|---|---:|---:|---:|---:|---:|
${seedRows}
| 均值 | ${format(last.meanBits.motion)} | ${format(last.meanBits.feedback)} | ${format(last.meanBits.attention)} | ${format(main.sameEvents.mean)} | ${format(secondary.sameEvents.mean)} |

${comparisonText('主比较：运动减注意力', main)}

${comparisonText('次比较：原反馈减注意力', secondary)}

只有三个种子，上述区间很不稳定；应结合各个种子的方向与独立复现判断。

## 短程与长程的区别

预先安排的 1,024 事件短程运行中，注意力相对运动平均改善 ${format(screen.comparisons.vsMotion.sameEvents.mean)}，相对原反馈改善 ${format(screen.comparisons.vsFeedback.sameEvents.mean)} bit/事件。长程不更改模型或超参数。下表展示全部预设检查点，不能只挑中途最佳值来替代最终比较。

| 训练事件 | 运动均值 | 原反馈均值 | 注意力均值 | 运动减注意力 | 原反馈减注意力 |
|---|---:|---:|---:|---:|---:|
${curveRows}

## 注意力是否真的在工作

以下为每种子最多 32 个开发事件的诊断；平均增益 1 表示中性。最后一列是在训练完的同一个模型中，推理时关闭全部视觉门控后的损失变化，正值表示关掉后更差。

| 种子 | 亮度平均增益 | 帧差平均增益 | 学习运动平均增益 | 关闭门控增加的损失 |
|---|---:|---:|---:|---:|
${diagnosticRows}

三种子的关闭门控损失变化均值为 ${format(mean(result.diagnostics.map(x => x.attention.disabledMinusEnabledBits)))} bit/事件。该干预可能让已经依赖门控的模型偏离训练时状态，不能等同于与从头训练的无门控模型比较，更不能据此证明注意力总体有效。

## 验证、限制与下一步

自动测试 11 项全部通过，覆盖有限差分梯度、跨通道归一化梯度、前缀因果性、门控均值、模型读回、阻断缓存及训练状态恢复。原有编码器的部署编号保持兼容，新增编码器可通过命令行选择。

本轮对齐事件数，未建立相同计算预算结论。归一化把增益的理论范围由 [0.75,1.25] 改为 [0.6,5/3]，本轮无法拆分竞争与增益范围变化的贡献。数据是已多次使用的开发场景，不是独立正式测试；也没有专门验证干扰物过滤或空间注意能力。

建议优先检查训练曲线波动，并用新种子验证，再考虑空间注意力与干扰物任务。${conclusion}

完整配对数据及诊断见 [比较 JSON](${relativeComparison})；每个模型目录还保存协议、检查点、部署模型和可恢复的训练状态。
`;
fs.writeFileSync(outputFile, text, {flag:'wx'});
console.log(outputFile);
