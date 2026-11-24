# 预测性连接生长实验

## Hebb 与两端反馈选边

新版入口为 `synaptic_experiment.js`，使用 Node.js 22，无须安装第三方包。实验设计见 [学习依赖的突触形成、稳定与修剪](brain_inspired_synaptic_experiment.md)。 2026-11-23 的正式 A 和记忆试验见 [实验结果](synaptic_experiment_results_2026-11-23.md)。

32 个单元从训练开始就接收同样的视觉输入。各组共享初始的 128 条循环边和共同权重。两端选边用源单元上一帧活动乘以目标单元的有符号反馈，累计缺失边的收益估计。Hebb 组使用先后激活的正相关，联合组用相关性加权反馈排名。每 64 个事件最多试接 4 条边，其中 1 条来自随机探索。后续事件用冻结在同一时刻的权重比较开关该批边的损失，连续窗口的结果决定成熟或撤回。

复现三种子试跑及其连接回放：

```powershell
node .\synaptic_experiment.js --seeds 3 --train 1024 --samples 256 --checkpoints 0,128,512,1024 --compute-match --output .\runs\synaptic-pilot-replay
node .\plot_synaptic_experiment.js .\runs\synaptic-pilot-replay
```

在浏览器打开该运行目录下的 `report.html`。滑动生长轮次可以追踪真实选中的端点、评分、成熟和撤回。报告中的 `summary.json` 汇总各方法，每个种子还保存自己的检查点、阻断结果、连接历史及可加载模型。`protocol.json` 在运行开始前保存参数和源代码哈希。`--compute-match` 给出同累计估算计算量的对照，包括候选评分和逐批阻断的开销。这项计数是估算值，另报耗时和采样的进程 RSS。RSS 包括同进程运行的其他方法和数据。

默认 `motion` 编码器以因果方式计算归一化池化像素、减背景后的纵向质心、从第一帧到当前帧的平均位移速度及最近一帧的位移。所有视觉量都从像素计算。`--encoder pooled` 可以只使用池化像素和同步音频。环境先均匀抽取接触帧所在的时间区间，再从该区间构造初速，并逐帧检测球下缘与地面的首次接触。用独立解析解复核接触帧。每次训练取两个早期前缀和一个较晚的撞击前前缀，收到声音反馈后才计算损失。

使用三个独立数据档位。`pilot` 用于探索。`development` 默认检查 0、128、512、2048、8192，另取独立开发场景。`formal` 使用独立的场景种子命名空间，默认 10 个训练种子、8,192 个训练事件和每种子 2,000 个测试事件。固定容量对照的默认 176 条循环边是起步值，运行前可通过 `--fixed-capacity` 配置。可用 `--rate` 登记学习率。 `--workers N` 可按种子并行运行，默认 1。运行协议记录工作线程数。测量单进程耗时时使用 `--workers 1`，并行运行的 RSS 包含所有工作线程。

A→B→A 保持实验与局部反馈对照可通过以下命令运行。`--train` 在记忆试验中表示每个阶段的训练事件数。

```powershell
node .\synaptic_experiment.js --memory --train 256 --seeds 3 --samples 128 --methods fixed,random,joint,local_fixed,local_random,local_joint --output .\runs\synaptic-memory
```

`local_*` 使用固定随机反馈和局部资格迹。实现以反向方式计算与显式前向资格迹相同的标量递推，仍保留活动历史，用于验证学习规则。`--trace-decay 0` 关闭跨帧痕迹。`joint_no_prune`、`joint_no_stability` 和 `joint_no_homeostasis` 可加入 `--methods` 做消融。

运行 `node --test test_synaptic_experiment.js test_predictive_growth.js` 验证物理接触、前缀因果性、有限差分梯度、候选评分、局部资格迹等价性、连接状态变化和旧版入口。

## 竞争式通道注意力

`--encoder hierarchical-attention` 在可学习运动基础上，用上一帧高层状态产生七个视觉通道的正增益，再除以通道均值。各通道平均增益保持为 1，因此会相对竞争；声音不受门控。相比原反馈模型不增加参数。这是特征通道注意力，不是空间位置注意力。

开发方案、收益门槛和已知边界见 [注意力实验设计](attention_design_2026-11-24.md)，实测结果见 [注意力开发结果](attention_development_2026-11-24.md)。运行新实验时请使用空输出目录：

```powershell
node .\synaptic_experiment.js --profile development --encoder hierarchical-attention --methods fixed --train 8192 --seeds 3 --samples 256 --output .\runs\my-attention-development
```

分别运行同配置的 `hierarchical-motion` 和 `hierarchical-feedback` 后，可汇总配对结果：

```powershell
node .\compare_attention.js MOTION_DIR FEEDBACK_DIR ATTENTION_DIR OUTPUT_JSON
```

## 第一轮实验

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

## 分层视觉

开发比较和已知限制见 [分层视觉开发记录](hierarchical_development_2026-11-24.md)。

`--encoder hierarchical` 使用固定、共享的局部视觉滤波；`--encoder hierarchical-delta` 另外加入相邻帧差。输入先以 2×2 均值降为 16×16，再以边缘复制的 3×3 邻域计算亮度、中心—周围 ON/OFF、水平和竖直差分，最后以 2×2 均值聚合成每通道 8×8。首帧的时间差为零。同期声音作为独立输入。高层仍是原来的 32 单元循环网络，反馈关闭。滤波器无可训练参数，高层输入投影可训练；各层参数数目写入检查点。协议记录具体边界与时间约定。

```powershell
node .\synaptic_experiment.js --profile development --encoder hierarchical --methods fixed --output .\runs\hierarchical-development
node .\synaptic_experiment.js --profile development --encoder hierarchical-delta --methods fixed --output .\runs\hierarchical-delta-development
```

`hierarchical-motion` 在帧差上增加共享、可学习的 3×3 卷积和 tanh；`hierarchical-feedback` 再用上一帧高层状态计算各通道增益 `1 + 0.25*tanh(Wa)`，首帧恒为 1。两者共享相同初始化和参数尺寸；前者不启用反馈矩阵，因此反馈矩阵在控制组中不参与训练。它们目前只支持 BPTT。固定低层滤波无可训练参数；中层卷积、反馈、高层投影、循环边和读出分别计数。最终报告还在最多 32 个开发场景上记录各输入通道的非零率、空间覆盖、跨帧变化和整通道遮蔽效果。

每个种子的部署模型、训练状态数据和审计历史分别保存。可在程序中读取 `<method>-training-state.json` 与 `<method>-edges.json`，调用 `SynapticNetwork.restore(state, audit)` 恢复并继续调用 `train(event)`；调用方负责提供正确的后续事件。此恢复接口适用于包含 `workHistory` 的新版状态文件。估算计算量包括分层滤波与网络运算，但仍只是近似计数；场景生成不计入。跨架构按相同种子和开发数据分别运行，再用 `node .\compare_hierarchies.js CONTROL_DIR TREATMENT_DIR` 比较同事件数及共同预算内最近保存检查点。后者受检查点间距限制；不同编码器间仍需谨慎解释计算量估算。

设计要求先验证固定结构的开发门槛，再开展后续结构比较。可学习运动和反馈的梯度、因果性、模型读回和训练恢复已通过自动测试；反馈是否改善预测需要配对开发实验检验。尚未实现与非分层基线具有相同有效可训练参数数目的对照，也未完成独立正式测试。
