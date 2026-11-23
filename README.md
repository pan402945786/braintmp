# 预测性连接生长实验

`predictive_growth_experiment.js` 不依赖第三方包，实现了
`predictive_growth_experiment.md` 中描述的第一轮实验。

主要内容包括：

- 可通过随机种子复现的落球场景生成器：每段 48 帧，画面为 32×32 灰度图，并带有同步撞击声；
- 稀疏循环预测网络，记录连接的出生时间、稳定状态、贡献和删除信息；
- 在事件结束后，使用保存的前缀进行在线学习；
- 时间先验、固定初始连接、随机选边生长、经验选边生长和固定容量对照组；
- 各方法共享测试事件和随机选取的观察截止帧；
- 负对数似然（单位：比特）、中位绝对帧误差、提前预测准确率曲线、成熟连接阻断效果、模型大小和预估压缩回本点。

运行一次快速试验：

```powershell
node .\predictive_growth_experiment.js `
  --train-events 512 `
  --dev-events 128 `
  --test-events 256 `
  --checkpoints 0,128,512 `
  --output .\predictive_growth_results.json
```

正式运行时，可按实验文档中登记的规模增加事件数量，例如
`--train-events 8192 --test-events 2000`。各方法使用相同的数据种子和每个事件的观察截止帧，因此可以进行配对比较。

也可以将实验作为模块导入：

```js
const { SceneGenerator, SparseGrowingRNN, runExperiment } = require('./predictive_growth_experiment');
const report = runExperiment({ trainEvents: 128, testEvents: 64 });
```

进行多随机种子比较时，可以调用 `runStudy({ seeds: 10, ... })`。它会返回每个种子的报告，以及各组损失和随机生长与经验生长之间差值的配对 bootstrap 95% 置信区间。

默认配置适合快速试跑，便于在只安装了 Node.js 的环境中验证流程。在得出科学结论之前，应使用登记的实验规模重新运行并分析结果。
