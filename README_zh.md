# rebend-rl

面向动态柔性作业车间重调度的强化学习引导逻辑 Benders 分解研究仓库。

English version: [README.md](README.md)

## 项目概述

本仓库基于 `lbbd_rl_unroll_proposal.tex` 中的研究方案搭建，核心目标是研究一种用于 **动态柔性作业车间重调度** 的混合求解框架，将以下三部分结合起来：

- 结构化的 **逻辑 Benders 分解（LBBD）**；
- 由 **CP-SAT / CP / 小规模 MIP** 执行的局部精确排程子问题；
- 一个 **强化学习策略**，它不直接生成完整排产方案，而是控制求解器每一轮如何展开搜索。

目标问题是带扰动的 **DFJSP-Repair**：在机器故障、新订单到达、加工时间变化、混合扰动等场景下，快速恢复可行性，并在以下目标之间取得平衡：

- 生产效率；
- 计划稳定性；
- 响应时间。

## 当前仓库状态

目前仓库中主要包含开题方案源文件和 PDF：

- `lbbd_rl_unroll_proposal.tex`
- `lbbd_rl_unroll_proposal.pdf`

也就是说，这个仓库现在更像一个“研究启动点”，后续实现将按实验阶段逐步补齐。

## 现状回顾：这些实验现在并没有都跑完

没有。就“是否已经完成支撑顶会投稿的完整实验包”这个问题而言，当前仓库还**没有**到那个阶段。

目前已经具备的内容主要是：

- 研究问题定义与方法提案；
- 一套相对完整的实验蓝图；
- 一个 RL 引导局部 repair 的最小可复现实验原型。

但以下这些关键实验块，**还没有**以论文级规模全部完成：

- 在统一预算下覆盖 **小/中/大规模** 的系统实验；
- 覆盖精确优化、分解方法、元启发式、现代神经调度器的 **更完整 baseline 套件**；
- 跨规模、柔性比、扰动类型、benchmark family 的 **泛化实验**；
- 组件级、表示级、奖励级、训练级的 **消融实验**；
- **公开 benchmark / 更真实 benchmark** 上的评测，如 Brandimarte、Hurink 等；
- 至少 `5` 个训练种子、置信区间、paired significance test 等 **统计稳健性报告**；
- 求解器后端迁移、长预算 exactness、等算力公平性等 **审稿防御实验**。

所以更准确的说法是：当前仓库是一个**朝顶会实验标准推进的研究仓库**，而不是“这些实验已经全部跑完的定稿仓库”。

## 参考相关论文后，应满足的实验标准

下面的实验设计，不是凭空拍脑袋列出来的，而是参考了这类调度/强化学习论文里已经比较稳定的实验范式，尤其包括：

- **L2D**：*Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning*（2020），它基本确立了“要和人工 dispatching rule 对比，并测试对更大未见实例的泛化”这一套实验预期；
- **DAN**：*Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning*（2023），进一步强化了“合成数据 + 公共 FJSP benchmark”双线验证的重要性；
- **DRL-Guided Improvement Heuristic**（ICLR 2024），把 improvement-based RL 与更强局部搜索对比的门槛抬高了；
- **Graph Neural Networks for Job Shop Scheduling Problems: A Survey**（2024），系统总结了现代调度论文越来越强调 benchmark 覆盖、baseline 公平性、泛化能力和统计严谨性。

也正因为如此，下面的“实验计划”不应被理解为可选项，而应被理解为：**如果要支撑顶会投稿，这些实验块基本都得补齐。**

## 方法概要

本项目拟研究的是一个 **展开式（unrolled）LBBD 求解器**：

1. **主问题** 负责高层修复决策，例如机器分配、冻结/解冻和修复区域选择；
2. **子问题** 在当前修复区域内执行精确或近精确排程；
3. 求解过程中逐步加入 **可行性割、关键路径割、最优性割、稳定性相关割**；
4. RL 策略观察当前求解状态，并决定：
   - 修哪里；
   - 修多大；
   - 子问题给多少预算；
   - 后续版本里还可以决定优先加入哪些割。

关键思想不是“让神经网络直接排产”，而是“让学习方法控制结构化求解过程”。

## 实验计划

下面这部分不是轻量级的项目说明，而是按顶会主实验章节标准整理的实验蓝图，目标是让实验设计本身足够经得起严苛审稿。

### 1. 实验目标与核心验证点

本方案需要围绕五个核心主张来设计实验，而不是围绕代码模块罗列结果。

1. **anytime 优势**：在短到中等在线预算下，RL 引导的 LBBD 相比 plain LBBD 与人工修复规则，能更快找到更优修复解。
2. **结构化学习优势**：学习方法若用于“控制求解器搜索过程”而不是“直接生成完整排程”，在可行性恢复与稳定性保持上更有优势。
3. **泛化能力**：学到的控制策略应能跨实例规模、柔性度、扰动组合和基准族泛化，而不是只记住训练分布。
4. **鲁棒性与精确性兼容**：学习模块改善有限预算下的搜索顺序，而 fallback LBBD 保留长预算下的完备性与精确性恢复能力。
5. **性能来源可归因**：性能提升应能清晰归因到修复区域选择、预算控制、割优先级、模仿预训练、异构图编码等具体组件。

审稿人最关心的不是“比谁高了几点”，而是“你到底验证了什么命题，证据是否闭环”。因此实验章节必须围绕这五点组织。

### 1.5. 支撑顶会投稿的最小实验包

如果目标是支撑一篇严肃的 NeurIPS / ICLR / ICML / OR 风格论文，下面这些实验块基本都要落地，不能只做其中一小部分。

| 实验块 | 至少要跑什么 | 审稿人为什么一定会问 |
|---|---|---|
| 规模梯度实验 | 小/中/大规模，在统一在线预算下系统比较 | 防止只挑一个对自己有利的规模 |
| 广泛 baseline | 精确法、LBBD、启发式、元启发式、神经方法 | 防止对比面过窄 |
| 公共 benchmark | Brandimarte、Hurink 及相关家族 | 证明和主流文献真正接轨 |
| 动态重调度集 | 静态 benchmark 转 repair 任务 + 合成动态集 | 证明研究的是 rescheduling，而不只是静态排程 |
| 泛化实验 | ID、Scale-OOD、Flexibility-OOD、Disturbance-OOD、Family-OOD | 支撑“不是只记住训练分布” |
| 消融实验 | 组件、表示、奖励、训练协议多维消融 | 说明性能提升到底来自哪里 |
| 统计稳健性 | 至少 `5` 个种子、CI、paired test、effect size | 防止单次运行过度宣称 |
| 审稿防御实验 | 等算力、公平性、长预算 exactness、后端迁移 | 为 rebuttal 提前准备证据 |

凡是明显缺少这张矩阵中大块内容的结果，都更适合表述为**原型验证**或**阶段性实验**，而不是“完整论文证据”。

### 2. 数据集与预处理

建议使用三类互补数据源，分别支撑标准对比、可控扩展和泛化结论。

#### A. 公共静态基准转动态 repair 任务

选用调度文献中广泛使用的 JSSP/FJSP 基准族：

- **Brandimarte (MK)**：FJSP 主基准；
- **Hurink**：覆盖不同柔性水平；
- **Dauzere-Peres / Paulli**：作为额外 FJSP 泛化测试；
- **Taillard / Lawrence / ABZ 风格 JSSP**：可作为低柔性甚至零柔性的特例压力测试。

处理流程如下：

1. 用强求解器离线求解原始静态实例，得到高质量初始排产 `x0`；
2. 将 `x0` 视为基准生产计划；
3. 在时刻 `t_d` 注入扰动，生成动态重调度任务；
4. 所有方法都从相同的 `x0 + disturbance` 状态出发比较。

其中，初始排产建议用以下流程生成：

- CP-SAT / CP Optimizer 大预算求解；
- 必要时用启发式 warm start；
- 对同一实例保存最优或最强 incumbent 作为统一起点。

#### B. 合成动态 FJSP 实例

为了系统控制难度与分布，建议构建合成实例生成器，覆盖：

- 工件数 `J in {10, 20, 50, 100}`；
- 机器数 `M in {5, 10, 20}`；
- 每个工件的工序数从 `Uniform{5, 20}` 采样；
- 机器柔性比 `f in {0.2, 0.5, 0.8}`；
- due-date tightness 取 `{1.2, 1.5, 2.0}`；
- 负载率从中等拥塞到高拥塞均覆盖。

扰动类型至少包含：

- **机器故障**：在随机时刻注入停机窗口，时长设为初始排程总时长的 `5%-20%`；
- **新订单到达**：在 `t_d` 后插入 `5%-30%` 额外工件；
- **加工时间膨胀**：随机抽取一部分工序，将加工时间乘以 `1.2-1.8`；
- **急单插入**：插入 due date 更紧、拖期惩罚更高的工件；
- **混合扰动**：至少两种扰动同时发生。

#### C. 跨分布泛化测试集

泛化实验不能只做“换随机种子”，而要显式设计分布迁移：

- **ID 测试**：与训练分布同尺度同扰动，但实例和扰动种子全新；
- **Scale-OOD**：训练只在 `{10,20}` 工件规模上，测试扩展到 `{50,100}`；
- **Flexibility-OOD**：训练在中等柔性，测试在极低/极高柔性；
- **Disturbance-OOD**：训练只见单一扰动，测试转到混合扰动；
- **Family-OOD**：训练只用合成数据，测试零样本迁移到公开 benchmark families。

#### 预处理与数据划分协议

- 合成实例按生成随机种子划分为 `80/10/10` 的训练/验证/测试集；
- 所有公开 benchmark family 都禁止参与训练和超参数选择，只用于最终测试；
- 时间、加工时长、due date 等特征按实例尺度做归一化，例如除以 `x0` 的 horizon 或实例统计量；
- 状态图中只允许使用重调度时刻可观测到的因果信息，严禁未来信息泄漏；
- 对每个静态基准实例生成 `K=10` 个测试扰动 realization，用于 paired comparison。

### 3. 基线方法选择

基线必须覆盖四个维度：精确优化、分解优化、传统启发式/元启发式、现代学习方法。否则审稿人会认为对比面不完整。

#### A. 优化与重调度基线

- **Full CP-SAT Rescheduling**：在相同在线预算下从零重排；
- **Full CP Optimizer / MIP**：用于小中规模强基线和上下界参考；
- **Rolling Horizon Repair**：工业界常见重调度思路；
- **ALNS / Ruin-and-Recreate**：强大邻域搜索基线；
- **Tabu Search / Simulated Annealing / GA**：代表性元启发式。

这组基线主要应对的审稿问题是：“你真的比成熟 OR 方法更强吗？”

#### B. 分解类基线

- **Plain LBBD**：固定 repair region、固定 cut management、固定子问题预算；
- **Critical-Path LBBD**：基于关键路径或关键机器块的人工区域扩张；
- **Random-Region LBBD**：随机 repair region；
- **Oracle-Region LBBD**：离线 hindsight 选择最佳 repair region，作为上界参考。

这组基线用于在相同算法骨架内隔离“学习控制”本身的价值。

#### C. 学习类基线

- **Dispatching Rules**：SPT、LPT、MWKR、EDD、CR 等；
- **L2D 风格 GNN dispatching**：代表构造式 RL 调度；
- **DAN 风格 FJSP RL**：代表操作-机器注意力式 FJSP 学习方法；
- **DRL-guided improvement heuristic**：代表改进式而非构造式 RL；
- **CP-guided RL dispatching**：代表 CP + RL 混合调度；
- **Dynamic RL rescheduling baseline**：代表已有面向在线到达或动态车间事件的 DRL 方法。

#### 基线公平性原则

- 所有方法使用**相同在线 wall-clock budget**；
- 所有学习方法使用**相同训练步数预算**和相同 early-stopping 规则；
- 所有非学习基线使用**相同调参预算**；
- 若某方法原生不支持稳定性目标，应显式说明如何把稳定性正则整合进其目标中；
- 所有结果都必须在同等资源条件下比较，不能出现“学习方法用 GPU，优化方法只给单线程 CPU，却只比 wall-clock”的不公平设定而不说明。

### 4. 评估指标

#### 主指标

1. **归一化修复目标值 `Obj_norm`**  
   即提案中的加权目标函数在实例级做归一化，便于跨尺度汇总。
2. **Anytime AUC**  
   统计 `{1s, 5s, 10s, 30s, 60s, 300s}` 上 objective-time curve 的面积。
3. **可行率**  
   给定时间预算内恢复可行解的比例。

这三个指标应出现在主表中，是论文的 headline metrics。

#### 辅助指标

- `C_max`；
- 总拖期；
- 平均开始时间偏移；
- 机器重分配率；
- 同机序列扰动距离；
- 首次找到可行解的时间；
- 相对最优参考或最强精确解的最终 gap；
- LBBD 迭代次数；
- 生成/接受的割数量；
- master 与 subproblem 的平均求解时间。

#### 统计稳健性要求

- 所有学习方法至少报告 `5` 个独立训练随机种子的 **mean ± std**；
- 所有测试结果同时报告 **95% 置信区间**；
- 方法间比较采用 **paired Wilcoxon signed-rank test** 或 **paired bootstrap**；
- 多重比较使用 **Holm-Bonferroni** 校正；
- 额外报告效应量，如 **Cliff's delta**。

只给单次结果或只给平均值，在顶会审稿里通常不够。

### 5. 实验设置与超参数

#### 软件栈

- PyTorch `2.x`
- CUDA `12.x`
- OR-Tools CP-SAT `9.x`
- IBM CP Optimizer `22.x`（若可用）
- Gurobi `11.x`（用于小中规模 MIP 参考）

#### RL 训练设置

- 算法：**PPO**
- 优化器：**AdamW**
- 学习率：搜索 `{1e-4, 3e-4}`，默认 `3e-4`
- value loss coefficient：`0.5`
- entropy coefficient：`{0.001, 0.01}`
- PPO clip ratio：`0.2`
- discount factor：`0.99`
- GAE lambda：`0.95`
- 梯度裁剪：`1.0`
- rollout horizon：`256`
- 每次更新 PPO epoch：`4`
- mini-batch size：`64`
- environment batch size：`32-64`

#### 模型设置

- 异构 GNN 编码器，`4` 层 message passing；
- hidden dim 取 `{128, 256}`；
- dropout 取 `{0.0, 0.1}`；
- anchor / type / radius / budget 分别使用独立策略头；
- 完整模型中再加入 cut-ranking head。

#### 模仿预训练设置

- expert pool 包括：critical-path-first、affected-machine-first、largest-delay-first、小规模 oracle best-improvement；
- 预训练样本量不少于 `100k` 状态-动作对；
- 训练 epoch `20-50`；
- 用独立 held-out trajectory 做验证。

#### 计算资源与复现要求

- 学习方法训练使用 `4-8` 张 A100 级别 GPU 或同级资源；
- 测试阶段所有方法在固定 CPU 线程预算下运行，例如每个 solver 使用 `16` 线程；
- 学习方法若在推理时使用 GPU，需要在论文中明确说明，并补充 CPU-only 推理结果；
- 公布所有随机种子、benchmark 生成脚本、扰动脚本、solver 参数文件、评测脚本。

#### 调参预算

- 学习方法：相同训练步数上限或相同 GPU-hour 上限；
- 元启发式/搜索基线：相同验证集调参预算，例如 `200` 次 Optuna trial 或等价 CPU-hour。

这部分是应对“对比是否公平”的第一道防线。

### 6. 主实验结果与预期分析

主论文建议至少包含以下四张核心表。

#### 表 1：ID 动态 FJSP 主结果

| Method | Obj_norm ↓ | Anytime AUC ↓ | Feasibility ↑ | Stability cost ↓ | Time-to-feasible ↓ |
|---|---:|---:|---:|---:|---:|
| Full CP-SAT |  |  |  |  |  |
| Plain LBBD |  |  |  |  |  |
| Critical-Path LBBD |  |  |  |  |  |
| ALNS |  |  |  |  |  |
| DAN-style RL |  |  |  |  |  |
| Proposed RL-LBBD |  |  |  |  |  |

#### 表 2：OOD 泛化结果

| Method | Scale-OOD Obj ↓ | Disturbance-OOD Obj ↓ | Flexibility-OOD Obj ↓ | Family-OOD Obj ↓ |
|---|---:|---:|---:|---:|
| Plain LBBD |  |  |  |  |
| DAN-style RL |  |  |  |  |
| Improvement RL |  |  |  |  |
| Proposed RL-LBBD |  |  |  |  |

#### 表 3：按扰动类型分解的结果

| Method | Breakdown ↓ | New arrivals ↓ | Proc.-time shift ↓ | Urgent jobs ↓ | Mixed ↓ |
|---|---:|---:|---:|---:|---:|

#### 表 4：效率与计算成本

| Method | Online budget | Mean runtime | GPU use | Training cost | Test cost / instance |
|---|---:|---:|---:|---:|---:|

#### 预期分析逻辑

- **Full CP-SAT / CP Optimizer** 可能在小规模、长预算下保持最强；
- **纯 RL dispatching** 可能速度快，但在可行性和稳定性上吃亏；
- **Plain LBBD** 会是非常强的结构化基线，但搜索顺序欠佳；
- **所提 RL-LBBD** 应主要在 anytime AUC、短预算可行率、稳定性约束下的修复质量上取胜，尤其是在中大规模和混合扰动场景中。

论文中不应把结论表述成“全面替代精确优化”，更合理、也更可信的主张应是：**在现实在线预算下，学习控制改善了结构化求解器的搜索效率。**

### 7. 消融实验设计

消融实验必须是递进式、可归因式的，而不是把若干开关随意拼在一张表里。

#### A. 组件级递进消融

- `A0`: Plain LBBD
- `A1`: `A0 +` learned region selection
- `A2`: `A1 +` learned budget control
- `A3`: `A2 +` learned cut prioritization
- `A4`: `A3 +` imitation pretraining
- `A5`: `A4 +` heterogeneous graph encoder
- `A6`: `A5 +` stability-aware features
- `A7`: `A6 +` gap-aware reward
- `A8`: `A7 +` fallback exact mode

#### B. 表示层消融

- 同构 GNN vs. 异构 GNN；
- 去掉 cut nodes；
- 去掉 disturbance nodes；
- 去掉历史轨迹特征；
- 去掉 critical-path indicator。

#### C. 学习机制消融

- PPO vs. imitation-only；
- 有无 expert policy KL regularization；
- dense reward vs. sparse terminal reward；
- 固定 subproblem budget vs. 自适应 budget action。

所有消融都应同时报告目标值与可行率，只报 objective 很容易掩盖“只是更激进但更容易失败”的情况。

### 8. 深入分析实验

想要经得住强审稿，不能只有主表和一张消融表，还需要更细的分析。

#### A. 参数敏感性分析

- 目标函数权重 `(alpha, beta, gamma, eta, kappa)`；
- repair radius 范围；
- cut-pool 大小；
- 最大 LBBD 迭代次数；
- 子问题时间预算。

#### B. 规模扩展分析

绘制性能随以下变量变化的曲线：

- 工件数；
- 机器数；
- 柔性比；
- 扰动强度。

#### C. 收敛与训练动态

- reward 随 environment step 的变化；
- validation objective 随训练 epoch 的变化；
- imitation warm start 对 sample efficiency 的影响；
- 不同随机种子下的训练方差。

#### D. 可视化与定性分析

- 可视化 repair region 的选择；
- 可视化 cut-pool 的演化；
- 对比策略是否仅追随 critical path；
- 展示代表性成功案例与失败案例。

#### E. 效率剖析

- master 与 subproblem 时间占比；
- performance profile / Dolan-More curve；
- 内存占用；
- 测试期 GPU 推理开销。

#### F. 统计显著性与稳健性

- 相对 plain LBBD 与最佳非学习基线做 paired significance test；
- anytime 曲线画置信区间带；
- 给出 win / tie / loss 统计。

### 9. 应对审稿人的补充实验预案

这一部分应在投稿前就准备好，而不是等 rebuttal 才慌忙补。

#### 攻击点 1：“提升只是因为你用了更多算力”

**防御性实验**：

- 所有方法使用相同在线 wall-clock budget；
- 所有方法使用相同调参预算；
- 额外报告离线训练成本；
- 与 imitation-only、oracle-region 等上界/弱化版本对比。

#### 攻击点 2：“策略不过是在学 critical-path heuristic”

**防御性实验**：

- 统计 repair region 与关键路径区域的重叠率；
- 去掉 critical-path 特征后重新测试；
- 用相同预算精调 strong critical-path LBBD 作为对照。

#### 攻击点 3：“学习模块破坏了 LBBD 的完备性”

**防御性实验**：

- 在小规模实例上做长预算实验；
- 打开 fallback 模式后，验证是否能收敛到与 plain LBBD 相同的最优/最强已知解；
- 单独汇报 no-regression feasibility 结果。

#### 攻击点 4：“方法只适用于你手造的 synthetic disturbance”

**防御性实验**：

- 零样本测试到公开 benchmark families；
- leave-one-disturbance-out 训练；
- 扰动强度 stress test；
- 若条件允许，加入一个具有相关性事件流的半真实模拟器。

#### 攻击点 5：“稳定性收益只是因为你调了目标权重”

**防御性实验**：

- 绘制效率-稳定性 Pareto front；
- 在 matched stability level 下比较方法；
- 报告多目标超体积指标。

#### 攻击点 6：“你的方法依赖特定求解器后端”

**防御性实验**：

- 选取一个代表性子集，同时用 CP-SAT 与 CP Optimizer 复现实验；
- 验证方法排序趋势是否保持一致。

#### 攻击点 7：“结果方差太大，不稳健”

**防御性实验**：

- 至少 `5` 个训练种子；
- 报告置信区间；
- 进行显著性检验；
- 给出跨实例族的 win-rate 表。

### 10. 实验局限性讨论

即使实验设计很强，也应主动承认经验范围的边界。

- 动态环境仍然部分依赖合成过程，无法完整覆盖真实 MES 级别的约束细节；
- 训练成本可能偏高，因为每个 episode 都要调用精确或近精确子问题；
- 某些强 OR 基线复现成本高，可能限制可完整覆盖的方法数量；
- 稳定性目前通过加权惩罚建模，尚未刻画人工排程员更细粒度的偏好；
- 尚未完全覆盖随机加工时长、能源目标、维护协同、多产线耦合等现实因素；
- 在公开 benchmark 上表现好，并不自动等价于真实工厂可直接部署。

## 近期建议的实现顺序

1. 先把扰动生成器和指标体系搭好；
2. 实现一个可复现的 plain LBBD baseline；
3. 加入 Brandimarte / Hurink 等公共 benchmark 的加载与转换；
4. 明确 RL-Region-LBBD 的状态、动作和奖励接口；
5. 把 baseline 套件扩展到 CP-SAT、ALNS、dispatching rule 与代表性 neural comparator；
6. 收集 expert 轨迹用于模仿预训练；
7. 接入 PPO 微调、多随机种子评测与 anytime 曲线；
8. 在做出更强论文结论之前，把泛化实验和消融矩阵补齐。

## README 使用说明

本 README 现在同时承担两个角色：

- 如实说明仓库当前做到哪里；
- 明确列出“若要支撑顶会投稿，还必须补哪些实验”。

因此阅读时需要始终区分两件事：

- **哪些内容已经实现或至少做过最小运行**；
- **哪些内容仍然属于论文级实验待办**。

## 说明来源

本 README 依据以下文件整理而成：

- `lbbd_rl_unroll_proposal.tex`
- `lbbd_rl_unroll_proposal.pdf`

## License

当前仓库尚未添加许可证。
