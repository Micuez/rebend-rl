# Top-Conference Experimental Plan for `rebend-rl` / `rebend-rl` 顶会级实验完整方案

> Version / 版本: v0.2  
> Date / 日期: 2026-05-18  
> Basis / 依据: `README.md`, `README_zh.md`, and the current minimal prototype artifacts under `experiments/results/latest_minimal_rl_region_lbbd/`

## 0. Venue and Compute Assumptions / 目标会议与算力假设

**中文**  
本版方案默认按 `NeurIPS/ICLR` 风格来组织，而不是按传统 OR 论文风格来组织。也就是说，重点会放在：清晰的问题定义、学习控制结构化求解器的算法新意、强泛化实验、严格消融、统计稳健性，以及一条能说服 ML 审稿人的因果证据链。  
同时，本版方案明确采用当前可用算力假设：**2 张 RTX 3090（`device0`, `device1`）**。因此，所有实验设计都必须区分：

- `full ambition`：理想的完整顶会证据链；
- `2x3090 feasible plan`：在当前算力下能真正推进的主计划。

**English**  
This version is organized explicitly in a `NeurIPS/ICLR` style rather than a traditional OR-paper style. The emphasis is therefore on a crisp problem definition, algorithmic novelty in learning to control a structured solver, strong generalization evidence, rigorous ablations, statistical discipline, and a causal evidence chain that can convince ML reviewers.  
This plan also explicitly assumes the currently available compute budget: **2 RTX 3090 GPUs (`device0`, `device1`)**. Accordingly, all experimental design should distinguish between:

- `full ambition`: the ideal complete top-conference evidence package;
- `2x3090 feasible plan`: the main executable plan under the current compute budget.

## 1. Project Positioning / 项目定位

**中文**  
`rebend-rl` 的核心目标不是训练一个直接生成完整排程的黑盒策略，而是研究一种“强化学习控制结构化求解器搜索过程”的混合框架：RL 负责决策修复区域、修复半径、子问题预算与后续 cut 优先级，LBBD + CP-SAT/CP/MIP 负责局部精确求解与全局可行性保障。项目面向的问题是动态柔性作业车间重调度（DFJSP-Repair），重点考察机器故障、新订单到达、加工时间变化和混合扰动下的快速 repair 能力。

**English**  
The core goal of `rebend-rl` is not to train a black-box policy that directly outputs schedules, but to study a hybrid framework in which reinforcement learning controls the search behavior of a structured solver. RL decides repair region, neighborhood size, subproblem budget, and later cut prioritization, while LBBD plus CP-SAT/CP/MIP handles local exact optimization and global feasibility recovery. The target problem is dynamic flexible job-shop rescheduling (DFJSP-Repair) under machine breakdowns, job arrivals, processing-time shifts, and mixed disruptions.

## 2. Honest Current Status / 当前真实状态

**中文**  
根据两个 README 与当前实验目录，仓库目前处于“研究构想明确、最小原型已跑通、顶会级完整实验尚未补齐”的阶段。当前已有：

- 方法提案与实验蓝图；
- 最小可复现实验配置 `experiments/configs/minimal_repro.yaml`；
- 一个最小 RL-guided region repair 原型；
- 最近一次原型结果中，`rl_guided_lbbd` 在该最小设置下优于 `plain_lbbd`、`critical_path_lbbd`、`random_region_lbbd` 和 `full_reschedule`。

**English**  
Based on the two READMEs and the current experiment artifacts, the repository is at the stage of having a clear research proposal and a minimally working prototype, but not yet a conference-grade empirical package. What already exists includes:

- the method proposal and experimental blueprint;
- a minimal reproducible config at `experiments/configs/minimal_repro.yaml`;
- a minimal RL-guided region-repair prototype;
- recent prototype evidence where `rl_guided_lbbd` outperforms `plain_lbbd`, `critical_path_lbbd`, `random_region_lbbd`, and `full_reschedule` under the current minimal setting.

## 3. Submission-Level Objective / 投稿级目标

**中文**  
本项目的目标优先收敛到一篇 **NeurIPS / ICLR 风格** 的论文，而不是同时兼顾过多 OR 叙事。核心论点应限定为：

1. 在有限在线预算下，RL 引导的 LBBD 能显著改善 anytime repair quality。
2. “学习控制求解器”比“纯神经直接排程”更适合动态 repair。
3. 方法能跨规模、柔性、扰动类型和 benchmark family 泛化。
4. 在长预算与 fallback 机制下，方法不牺牲 LBBD 的完备性叙事。
5. 性能提升能被清晰归因到可解释的算法组件。

**English**  
The submission-level goal is to support a paper in a **NeurIPS / ICLR style**, rather than trying to satisfy too many OR-style narratives at once. The paper should make only the following core claims:

1. RL-guided LBBD significantly improves anytime repair quality under limited online budgets.
2. Learning to control a solver is better suited to dynamic repair than pure neural direct scheduling.
3. The method generalizes across scale, flexibility, disturbance type, and benchmark family.
4. With long budgets and fallback mechanisms, the method does not sacrifice the completeness narrative of LBBD.
5. The performance gains can be attributed to interpretable algorithmic components.

## 4. Main Research Questions and Hypotheses / 核心科学问题与假设

| ID | 中文 | English |
|---|---|---|
| RQ1 | RL 是否能在短到中等 wall-clock budget 下改善 LBBD 的搜索顺序，从而更快找到更优 repair 解？ | Can RL improve the search order of LBBD under short-to-medium wall-clock budgets and therefore find better repair solutions earlier? |
| RQ2 | RL-LBBD 的优势是否主要体现在 feasibility、stability 与 anytime AUC，而非长预算最优性绝对统治？ | Is the advantage of RL-LBBD mainly in feasibility, stability, and anytime AUC rather than absolute domination in long-budget optimality? |
| RQ3 | 该控制策略是否能跨训练分布泛化，而不是只记住一个 instance generator？ | Can the control policy generalize beyond the training distribution rather than memorizing one instance generator? |
| RQ4 | 性能收益是否来自 region selection、budget control、cut prioritization、imitation pretraining 与 graph representation 等具体模块？ | Do the gains come from region selection, budget control, cut prioritization, imitation pretraining, and graph representation? |
| RQ5 | 在切换 solver backend 或改变扰动强度时，方法优势是否仍然保留？ | Does the method retain its advantage when the solver backend or disturbance severity changes? |

## 5. Paper Storyline / 论文叙事主线

**中文**  
论文叙事必须从“动态 repair 是一个受限时间下的结构化决策问题”出发，而不是从“我们训练了一个 RL 模型”出发。主线应是：纯 OR 方法在短预算下搜索顺序不够智能，纯神经方法在可行性和稳定性方面不够可靠，因此最合理的切入点是让学习模块控制结构化 solver 的展开方式。

**English**  
The paper narrative must start from dynamic repair as a structured decision problem under strict time budgets, not from “we trained an RL model.” The main story should be: pure OR methods are not search-efficient enough under short budgets, pure neural methods are not reliable enough in feasibility and stability, and therefore the right approach is to let learning control how a structured solver unfolds.

### 5.1 NeurIPS/ICLR Framing Rules / NeurIPS/ICLR 叙事约束

**中文**  
如果目标是 `NeurIPS/ICLR`，正文叙事要刻意避免写成“一个求解器工程堆料报告”。更合适的 framing 是：

1. 把方法定义成 **learning-to-control combinatorial search**，而不是单纯的 scheduling heuristic。
2. 把贡献点落在 **solver-state representation、budgeted decision making、generalization across disturbances、search-process credit assignment**。
3. 把主实验组织成“主张验证”，而不是“baseline 罗列表演”。
4. 把公开 benchmark、OOD、ablation、fairness 和 statistical rigor 放在与主结果同等重要的位置。
5. 少承诺“全局最优性超越”，多强调“practical budget 下的 anytime superiority + structured reliability”。

**English**  
If the target is `NeurIPS/ICLR`, the paper should avoid reading like an engineering-heavy solver report. A better framing is:

1. Define the method as **learning to control combinatorial search**, not merely as another scheduling heuristic.
2. Place the contributions around **solver-state representation, budgeted decision making, generalization across disturbances, and search-process credit assignment**.
3. Organize the main experiments around claim validation rather than baseline enumeration.
4. Treat public benchmarks, OOD tests, ablations, fairness, and statistical rigor as first-class evidence.
5. Avoid overpromising global optimality superiority; emphasize **anytime superiority under practical budgets plus structured reliability**.

## 6. Complete Experimental Matrix / 完整实验矩阵

| Block | 中文要求 | English requirement |
|---|---|---|
| Size ladder | 小/中/大规模统一预算比较 | Compare small, medium, and large scales under matched budgets |
| Public benchmarks | Brandimarte、Hurink、补充 FJSP/JSSP family | Brandimarte, Hurink, plus additional FJSP/JSSP families |
| Dynamic synthetic suite | 多规模、多柔性、多负载、多扰动强度 | Multi-scale, multi-flexibility, multi-load, multi-severity synthetic suite |
| Generalization | ID、Scale-OOD、Flexibility-OOD、Disturbance-OOD、Family-OOD | ID, Scale-OOD, Flexibility-OOD, Disturbance-OOD, Family-OOD |
| Baselines | 精确法、分解法、启发式、元启发式、学习法 | Exact, decomposition, heuristic, metaheuristic, and learning baselines |
| Ablations | 组件、表示、奖励、训练协议 | Component, representation, reward, and training-protocol ablations |
| Statistical rigor | 5 个以上训练种子、CI、paired tests、effect size | At least 5 training seeds, CIs, paired tests, effect sizes |
| Reviewer backup | 等算力、公平性、长预算 exactness、后端迁移 | Equal-compute, fairness, long-budget exactness, backend portability |

### 6.1 Priority Order Under 2x3090 / 2x3090 下的优先级重排

**中文**  
在只有 2 张 3090 的前提下，这个矩阵不能一次性全开，必须分层推进：

1. 第一优先级：`ID + 小中规模 + 分解类基线 + 少量学习基线 + 核心消融`。  
2. 第二优先级：`Scale-OOD + Disturbance-OOD + 公共 benchmark 子集`。  
3. 第三优先级：`Family-OOD 全量、solver-backend portability、最重的 reviewer backup`。

这意味着第一版投稿策略更适合做成：**问题定义清晰、主张集中、证据闭环强，但规模矩阵经过算力裁剪**。

**English**  
With only 2 RTX 3090 GPUs, this matrix cannot be fully expanded at once and must be layered:

1. First priority: `ID + small/medium scale + decomposition baselines + a small set of learning baselines + core ablations`.
2. Second priority: `Scale-OOD + Disturbance-OOD + a subset of public benchmarks`.
3. Third priority: `full Family-OOD, solver-backend portability, and the heaviest reviewer-facing backup experiments`.

That means the first submission strategy should likely be: **clear problem definition, focused claims, and a strong closed evidence loop, with the full matrix trimmed to match available compute**.

## 7. Datasets and Split Design / 数据集与划分设计

### 7.1 Public Benchmarks to Dynamic Repair / 公共静态基准转动态 repair

**中文**  
选取 Brandimarte (MK)、Hurink、Dauzere-Peres / Paulli，以及 Taillard / Lawrence / ABZ 风格实例。对每个静态实例，先用强离线求解器获得基准排产 `x0`，再在重调度时刻 `t_d` 注入扰动，构造 paired dynamic repair 任务。每个基准实例至少生成 `K=10` 个测试扰动 realization。

**English**  
Use Brandimarte (MK), Hurink, Dauzere-Peres / Paulli, and Taillard / Lawrence / ABZ-style instances. For each static instance, first obtain a strong baseline schedule `x0` with an offline solver, then inject disturbances at rescheduling time `t_d` to create paired dynamic repair tasks. Generate at least `K=10` disturbance realizations per benchmark instance.

### 7.2 Synthetic Dynamic DFJSP Suite / 合成动态 DFJSP 套件

| Dimension | 中文 | English |
|---|---|---|
| Jobs | `J in {10, 20, 50, 100}` | `J in {10, 20, 50, 100}` |
| Machines | `M in {5, 10, 20}` | `M in {5, 10, 20}` |
| Ops per job | `Uniform{5, 20}` | `Uniform{5, 20}` |
| Flexibility | `f in {0.2, 0.5, 0.8}` | `f in {0.2, 0.5, 0.8}` |
| Due-date tightness | `{1.2, 1.5, 2.0}` | `{1.2, 1.5, 2.0}` |
| Utilization | 中拥塞到高拥塞 | Moderate to high congestion |

扰动类型 / Disturbance types:

- 机器故障 / machine breakdowns
- 新订单到达 / new job arrivals
- 加工时间膨胀 / processing-time inflation
- 急单插入 / urgent job insertion
- 混合扰动 / mixed disturbances

### 7.3 Train/Val/Test and OOD Protocol / 训练验证测试与 OOD 协议

**中文**  
合成数据按 generator seed 做 `80/10/10` 划分；公开 benchmark 严禁参与训练与调参；训练只允许使用重调度时刻的因果可观测特征；OOD 测试必须显式包括 Scale-OOD、Flexibility-OOD、Disturbance-OOD 与 Family-OOD，不能仅用“换随机种子”代替泛化实验。

**English**  
Split synthetic data by generator seed into `80/10/10` train/validation/test sets. Public benchmarks must be excluded from training and hyperparameter tuning. Only causally observable features at rescheduling time may be used. OOD evaluation must explicitly include Scale-OOD, Flexibility-OOD, Disturbance-OOD, and Family-OOD rather than merely changing random seeds.

## 8. Baseline Suite / 基线套件

### 8.1 Optimization and Rescheduling Baselines / 优化与重调度基线

- Full CP-SAT Rescheduling
- Full CP Optimizer or MIP reference
- Rolling Horizon Repair
- ALNS or Ruin-and-Recreate
- Tabu Search / Simulated Annealing / GA

### 8.2 Decomposition Baselines / 分解类基线

- Plain LBBD
- Critical-Path LBBD
- Random-Region LBBD
- Oracle-Region LBBD

### 8.3 Learning Baselines / 学习类基线

- Dispatching rules: SPT, LPT, MWKR, EDD, CR
- L2D-style GNN dispatching
- DAN-style FJSP RL
- DRL-guided improvement heuristic
- CP-guided RL dispatching
- Dynamic RL rescheduling baseline from prior work

### 8.4 Fairness Rules / 公平性规则

**中文**  
所有方法共享相同在线 wall-clock budget；所有学习方法共享相同训练步数或 GPU-hour 预算；所有非学习方法共享相同调参预算；若某基线不原生支持稳定性项，必须清楚写出如何适配；CPU 线程数、GPU 使用方式、solver 参数与 early stopping 规则都必须公开。

**English**  
All methods must share the same online wall-clock budget. All learning methods must share the same training-step or GPU-hour budget. All non-learning baselines must share the same tuning budget. If a baseline does not natively support stability, its adaptation must be stated explicitly. CPU thread budgets, GPU usage, solver parameters, and early-stopping rules must all be disclosed.

### 8.5 Baseline Scope Recommended for 2x3090 / 2x3090 下建议保留的基线范围

**中文**  
在当前算力下，不建议第一轮就把所有学习类 baseline 全部重做。建议分为：

- `must-have`：Plain LBBD、Critical-Path LBBD、Random-Region LBBD、Full CP-SAT Rescheduling、Dispatching Rules。
- `strong optional`：ALNS 或 Ruin-and-Recreate 二选一。
- `paper-strengthening but expensive`：L2D-style、DAN-style、DRL-guided improvement heuristic 中最多选 `1-2` 个最相关代表。

原因很简单：对 `NeurIPS/ICLR` 审稿人来说，**一个做扎实的 learning-to-control 主线**，通常比“每种神经调度器都复现一点，但都不够扎实”更有说服力。

**English**  
Under the current compute budget, I do not recommend reproducing every learning baseline in the first round. Instead:

- `must-have`: Plain LBBD, Critical-Path LBBD, Random-Region LBBD, Full CP-SAT Rescheduling, and dispatching rules.
- `strong optional`: choose one of ALNS or Ruin-and-Recreate.
- `paper-strengthening but expensive`: pick at most `1-2` of L2D-style, DAN-style, and DRL-guided improvement heuristic as the most relevant neural comparators.

The reason is simple: for `NeurIPS/ICLR` reviewers, **a deeply validated learning-to-control story** is often more convincing than a shallow reproduction of every neural scheduler.

## 9. Metrics and Statistical Protocol / 指标与统计协议

### 9.1 Primary Metrics / 主指标

| Metric | 中文 | English |
|---|---|---|
| `Obj_norm` | 归一化 repair 目标值 | Normalized repair objective |
| Anytime AUC | 目标值-时间曲线面积 | Area under objective-time curve |
| Feasibility rate | 预算内恢复可行解比例 | Fraction of instances repaired feasibly within budget |

### 9.2 Secondary Metrics / 辅助指标

- Makespan / `C_max`
- Total tardiness
- Average start-time deviation
- Machine reassignment rate
- Sequence perturbation distance
- Time to first feasible solution
- Final optimality gap
- Number of LBBD iterations
- Number of generated and accepted cuts
- Master and subproblem solve time

### 9.3 Statistical Rigor / 统计稳健性

**中文**  
每个学习方法至少 `5` 个独立训练种子；报告 mean ± std、95% CI、paired Wilcoxon 或 paired bootstrap；多重比较使用 Holm-Bonferroni 校正；补充 Cliff's delta 或标准化配对效应量；所有主结论都必须能在 paired setting 下复核。

**English**  
Each learned method must use at least `5` independent training seeds. Report mean ± std, 95% confidence intervals, paired Wilcoxon or paired bootstrap tests, Holm-Bonferroni correction for multiple comparisons, and an effect size such as Cliff's delta. All main claims must be defensible under paired evaluation.

## 10. Method Variants to Run / 需要实际运行的方法版本

| ID | 中文 | English |
|---|---|---|
| M0 | Full CP-SAT / CP 基线 | Full CP-SAT / CP baseline |
| M1 | Plain LBBD | Plain LBBD |
| M2 | Critical-Path LBBD | Critical-Path LBBD |
| M3 | Random-Region LBBD | Random-Region LBBD |
| M4 | RL-guided region selection only | RL-guided region selection only |
| M5 | M4 + learned budget control | M4 + learned budget control |
| M6 | M5 + cut prioritization | M5 + cut prioritization |
| M7 | M6 + imitation pretraining | M6 + imitation pretraining |
| M8 | Full proposed model with heterogeneous graph encoding and fallback exact mode | Full proposed model with heterogeneous graph encoding and fallback exact mode |

## 11. Training Protocol / 训练协议

**中文**  
主训练算法采用 PPO，优化器使用 AdamW。起始搜索范围按 README：学习率 `{1e-4, 3e-4}`，entropy coefficient `{0.001, 0.01}`，PPO clip `0.2`，GAE `0.95`，discount `0.99`，rollout horizon `256`，mini-batch `64`。模型采用异构 GNN 编码器，消息传递层数建议 `4` 层，hidden dimension 从 `{128, 256}` 中选择。建议先做 imitation pretraining，再做 PPO fine-tuning。

**English**  
Use PPO as the main training algorithm with AdamW. Follow the README search ranges: learning rate `{1e-4, 3e-4}`, entropy coefficient `{0.001, 0.01}`, PPO clip `0.2`, GAE `0.95`, discount `0.99`, rollout horizon `256`, and mini-batch size `64`. Use a heterogeneous GNN encoder with around `4` message-passing layers and hidden size selected from `{128, 256}`. Imitation pretraining should precede PPO fine-tuning.

### 11.1 Compute-Constrained Training Plan for `device0,1` / 面向 `device0,1` 的训练计划

**中文**  
针对 2 张 RTX 3090，建议采用以下现实训练方案：

1. `device0` 负责主要训练任务，`device1` 用于并行评测、辅助训练或第二个种子。
2. 第一阶段只训练 `small` 与 `medium` 规模，不在训练阶段碰 `large`。
3. 用 `small -> medium` curriculum，而不是一上来混合全尺度训练。
4. 每轮只保留 `2-3` 个最关键超参数维度，避免大规模搜索。
5. 先做 imitation pretraining 缩短 PPO 冷启动时间。
6. 第一轮只做 `3` 个训练种子用于模型筛选，定版后再补到 `5` 个种子。
7. 若显存紧张，优先缩 batch size 和并行环境数，而不是先削模型表达能力。

**English**  
For 2 RTX 3090 GPUs, I recommend the following practical training setup:

1. Use `device0` for primary training and `device1` for parallel evaluation, auxiliary training, or a second seed.
2. Train only on `small` and `medium` scales in the first stage, and do not train on `large` yet.
3. Use a `small -> medium` curriculum instead of mixing all scales from the start.
4. Keep only `2-3` key hyperparameter dimensions active per sweep.
5. Use imitation pretraining to reduce PPO cold-start cost.
6. Start with `3` training seeds for model selection, then expand to `5` seeds once the setup is frozen.
7. If memory becomes tight, reduce batch size and the number of parallel environments before shrinking model expressiveness.

### 11.2 Recommended Compute Budget / 建议算力预算

| Item | 中文 | English |
|---|---|---|
| Training GPUs | `device0`, `device1` 两张 RTX 3090 | Two RTX 3090 GPUs: `device0`, `device1` |
| Primary training stage | 先小中规模，后 OOD | First small/medium, then OOD |
| Hyperparameter policy | 小网格或分阶段人工搜索 | Small grid or staged manual search |
| Seed policy | 筛选阶段 `3` seeds，定版后 `5` seeds | `3` seeds for screening, `5` after freezing |
| Neural baseline policy | 最多 `1-2` 个代表性神经基线 | At most `1-2` representative neural baselines |
| Large-scale protocol | 先只测试不训练 | Test-only before training |

## 12. Experimental Phases and Exit Criteria / 实验阶段与通过标准

| Phase | 中文目标 | English goal | Exit criterion / 通过标准 |
|---|---|---|---|
| P1 | 固化最小原型与日志协议 | Stabilize the minimal prototype and logging protocol | 所有 runs 可复现，结果文件齐全 |
| P2 | 建立 plain LBBD 与公共 benchmark loader | Build plain LBBD and public benchmark loaders | 能在公共实例上稳定运行 paired repair |
| P3 | 扩充动态扰动生成器与指标体系 | Expand disturbance generator and metric suite | 支持多扰动、paired evaluation、AUC 统计 |
| P4 | 跑完优化/分解/启发式第一轮基线 | Run first full baseline sweep | 至少得到小中规模完整基线表 |
| P5 | 建立 imitation pretraining 数据池 | Build imitation pretraining data | 至少 `100k` state-action pairs |
| P6 | 完成 PPO 多种子训练 | Complete multi-seed PPO training | 至少 `5` 个独立训练种子 |
| P7 | 完成 ID + OOD + ablation 主矩阵 | Complete ID, OOD, and ablation matrices | 主表、泛化表、消融表齐备 |
| P8 | 完成 reviewer backup 实验 | Complete reviewer-facing backup experiments | 等算力、长预算、后端迁移证据齐备 |
| P9 | 形成 paper-ready artifact | Produce paper-ready artifact package | 代码、日志、配置、图表、统计脚本可公开 |

### 12.1 Realistic Exit Criteria Under 2x3090 / 2x3090 下的现实通过线

**中文**  
如果按当前算力推进，第一版论文更现实的通过线应是：

1. 小中规模 ID 结果显著稳定优于 plain LBBD 和关键基线。
2. 至少完成 `Scale-OOD` 与 `Disturbance-OOD` 两类泛化实验。
3. 至少完成 `核心组件递进消融 + 表示消融`。
4. 至少有一个公开 benchmark family 子集结果。
5. 至少一个强学习基线或强改进式基线作为 neural comparator。

如果这些都没有，不建议把目标直接定成完整 `NeurIPS/ICLR` 投稿包。

**English**  
Under the current compute setup, a more realistic first-paper threshold is:

1. Stable and significant gains over plain LBBD and key baselines on small/medium ID tasks.
2. At least two generalization tracks: `Scale-OOD` and `Disturbance-OOD`.
3. At least `incremental core-component ablations + representation ablations`.
4. Results on at least one subset of a public benchmark family.
5. At least one strong neural or improvement-based comparator.

If these are not in place, I would not recommend calling it a full `NeurIPS/ICLR`-ready package yet.

## 13. Main Tables and Figures for the Paper / 论文主表与主图

**中文**  
主文建议至少包含 4 个主表和 5 组主图：

1. ID 主结果表：`Obj_norm`、Anytime AUC、Feasibility、Stability、Time-to-feasible。
2. OOD 泛化表：Scale-OOD、Flexibility-OOD、Disturbance-OOD、Family-OOD。
3. 扰动类型分解表：breakdown、arrival、processing-time shift、urgent jobs、mixed。
4. 效率与成本表：runtime、GPU cost、CPU cost、training cost、test cost，并明确 2x3090 训练设置。
5. Anytime 曲线图。
6. 规模扩展图。
7. 消融递进柱状图或折线图。
8. Pareto 前沿图：效率-稳定性权衡。
9. 定性可视化：repair region、critical path、cut-pool evolution。

**English**  
The main paper should contain at least four key tables and five figure groups:

1. ID main results table with `Obj_norm`, Anytime AUC, Feasibility, Stability, and Time-to-feasible.
2. OOD generalization table over Scale-OOD, Flexibility-OOD, Disturbance-OOD, and Family-OOD.
3. Disturbance-type breakdown table.
4. Efficiency and cost table, explicitly reporting the 2x3090 training setup.
5. Anytime curve figure.
6. Scaling figure.
7. Incremental ablation figure.
8. Pareto frontier figure for efficiency versus stability.
9. Qualitative visualization of repair regions, critical paths, and cut-pool evolution.

## 14. Reviewer-Risk Mitigation Plan / 审稿风险防御方案

| Risk | 中文应对 | English defense |
|---|---|---|
| Extra compute criticism | 做等算力和等 wall-clock 对比，单列训练成本 | Run equal-compute and equal wall-clock comparisons, and report training cost separately |
| “Just a critical-path heuristic” | 测 region overlap，去掉 critical-path features，再与 tuned critical-path baseline 对比 | Measure overlap, remove critical-path features, and compare to a tuned critical-path baseline |
| Exactness concern | 小规模长预算实验验证 fallback 后与 plain LBBD 一致收敛 | Use long-budget small-scale tests to show convergence consistency with plain LBBD under fallback |
| Synthetic overfitting | 零样本迁移到公开 benchmark，做 leave-one-disturbance-out | Test zero-shot transfer to public benchmarks and leave-one-disturbance-out training |
| Objective-weighting objection | 画 Pareto 前沿，在 matched stability level 下比较 | Plot Pareto frontiers and compare at matched stability levels |
| Solver-specific concern | 子集上切换 CP-SAT 与 CP Optimizer | Re-run a subset with CP-SAT and CP Optimizer |
| High variance concern | 五个以上种子、置信区间、paired significance tests、win/tie/loss 表 | Use 5+ seeds, CIs, paired significance tests, and win/tie/loss summaries |

## 15. Artifact and Reproducibility Requirements / 复现与交付要求

**中文**  
最终可投稿版本必须公开或内部归档以下内容：

- 数据生成脚本与 benchmark loader；
- 扰动注入脚本与 seed 列表；
- 所有训练与测试配置；
- solver 参数文件；
- 原始 metrics、trace、日志与环境信息；
- 统计检验脚本；
- 作图脚本；
- 失败案例与异常 run 记录；
- 复现实验说明文档。

**English**  
The final submission-ready package must publish or internally archive:

- data-generation scripts and benchmark loaders;
- disturbance injection scripts and seed lists;
- all training and evaluation configs;
- solver parameter files;
- raw metrics, traces, logs, and environment info;
- statistical testing scripts;
- plotting scripts;
- failure-case records and abnormal runs;
- a reproducibility guide.

## 16. What Counts as “Paper-Ready” / 什么才算“达到论文可投稿状态”

**中文**  
只有满足以下条件，才应把该项目表述为“具备顶会投稿级实验支撑”：

1. 公共 benchmark 与 synthetic suite 两条线都跑通。
2. 基线覆盖精确、分解、启发式、学习四大类。
3. 至少完成 5 个训练种子和成套统计检验。
4. 主结论在 ID 与至少三类 OOD 设置下同时成立。
5. 消融能解释主要收益来源。
6. reviewer backup 实验已提前准备。
7. 所有结果都有原始日志、配置和脚本可追溯。

**English**  
The project should be described as conference-ready only if all of the following hold:

1. Both public-benchmark and synthetic-suite tracks are complete.
2. Baselines cover exact, decomposition, heuristic, and learning families.
3. At least 5 training seeds and a full statistical package are complete.
4. Main conclusions hold in ID and at least three OOD settings.
5. Ablations explain the major gain sources.
6. Reviewer-facing backup experiments are already prepared.
7. Every result is traceable to raw logs, configs, and scripts.

## 17. Immediate Execution Roadmap for This Repository / 面向当前仓库的近期落地路线

| Priority | 中文 | English |
|---|---|---|
| 1 | 固化当前最小原型的 run contract、日志字段和评测脚本 | Freeze the run contract, logging schema, and evaluation scripts for the current minimal prototype |
| 2 | 先完成 plain LBBD 与 dynamic generator 的稳定版本 | Finish stable plain LBBD and dynamic generator implementations |
| 3 | 接入 Brandimarte / Hurink loader，形成公共 benchmark 线 | Add Brandimarte / Hurink loaders to establish the public benchmark track |
| 4 | 扩展 baseline 到 CP-SAT、ALNS、dispatching rules | Expand baselines to CP-SAT, ALNS, and dispatching rules |
| 5 | 做 imitation pretraining 数据采集和专家轨迹生成 | Collect expert trajectories for imitation pretraining |
| 6 | 启动 PPO 多种子训练与 anytime 曲线评测 | Launch multi-seed PPO training and anytime-curve evaluation |
| 7 | 跑 OOD 与消融矩阵 | Run OOD and ablation matrices |
| 8 | 补 reviewer backup 和最终图表统计 | Add reviewer backup experiments and final figures/statistics |

### 17.1 Recommended First Submission Scope / 建议的第一版投稿范围

**中文**  
结合 `NeurIPS/ICLR` 风格和 2x3090 现实条件，我建议第一版目标不要设成“全矩阵全跑完”，而是设成：

- 一个非常清晰的主命题：`RL improves LBBD search control for dynamic repair under practical budgets`；
- 一个扎实的小中规模主实验包；
- 两类 OOD；
- 一组强消融；
- 一组公共 benchmark 子集；
- 一套严谨统计；
- 一套对“是不是只是算力堆出来的”“是不是只是 critical-path heuristic”这两类核心质疑的正面防御。

这比“面面俱到但每块都偏薄”更像 `NeurIPS/ICLR` 会接受的第一版故事。

**English**  
Combining the `NeurIPS/ICLR` framing with the 2x3090 reality, I recommend that the first submission should not try to complete the entire matrix. Instead, aim for:

- one very clear thesis: `RL improves LBBD search control for dynamic repair under practical budgets`;
- a solid small/medium-scale main package;
- two OOD tracks;
- a strong ablation suite;
- one public-benchmark subset;
- rigorous statistics;
- direct defenses against the two most likely attacks: “it is just extra compute” and “it is just a critical-path heuristic.”

That is more aligned with what a first `NeurIPS/ICLR` submission can realistically support than trying to be exhaustive and ending up shallow everywhere.

## 18. Recommendation for Internal Review / 建议你的审核重点

**中文**  
你审核这份方案时，建议重点看四件事：  
第一，这个论文主张是否收得足够窄，避免过度宣称。  
第二，这个实验矩阵是否与你愿意投入的算力和时间匹配。  
第三，baseline 套件是否需要再收缩或再增强。  
第四，当前仓库更适合先冲“强原型论文”还是直接冲“完整顶会包”。

**English**  
When you review this plan, I recommend focusing on four questions.  
First, are the claims narrow enough to avoid overclaiming?  
Second, does the experiment matrix match the compute and time you are willing to invest?  
Third, should the baseline suite be narrowed or expanded?  
Fourth, is this repository better positioned first for a strong prototype paper or directly for a full top-conference package?

## 19. Closing Assessment / 总结判断

**中文**  
基于当前 README 所定义的方向，`rebend-rl` 很适合往 `NeurIPS/ICLR` 的 `learning-to-control structured search` 方向去讲，题目本身是成立的；但在 **2 张 RTX 3090（`device0`, `device1`）** 的现实约束下，最合理的策略不是贪大全，而是先做一个主张集中、证据闭环、统计扎实、OOD 与消融足够强的第一版投稿包，再决定是否继续扩成更完整的“全矩阵顶配版”。

**English**  
Based on the direction defined in the READMEs, `rebend-rl` is well suited to a `NeurIPS/ICLR` narrative around learning to control structured search. Under the realistic constraint of **two RTX 3090 GPUs (`device0`, `device1`)**, the most sensible strategy is not to be exhaustive immediately, but to build a first submission package with focused claims, a closed evidence loop, rigorous statistics, and sufficiently strong OOD and ablation support, and only then decide whether to expand it into a fully maximal evidence matrix.
