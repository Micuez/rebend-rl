# rebend-rl

RL-guided logic-based Benders decomposition for dynamic flexible job-shop rescheduling.

Chinese version: [README_zh.md](README_zh.md)

## Overview

This repository is the starting point for a research project based on the proposal in `lbbd_rl_unroll_proposal.tex`. The core idea is to combine:

- a structured **logic-based Benders decomposition (LBBD)** solver,
- a **local exact scheduling subproblem** solved by CP-SAT / CP / small MIP,
- and a **reinforcement learning policy** that does not directly output schedules, but instead controls how the solver expands its search.

The target problem is **dynamic flexible job-shop rescheduling (DFJSP-Repair)** under disturbances such as machine breakdowns, new job arrivals, processing-time changes, and mixed disruptions. The intended objective is to recover feasibility quickly while balancing:

- production efficiency,
- schedule stability,
- and limited response time.

## Repository Status

At the moment, this repository contains the proposal source and PDF:

- `lbbd_rl_unroll_proposal.tex`
- `lbbd_rl_unroll_proposal.pdf`

The implementation will be built around a staged experimental program rather than a single end-to-end release.

## Honest Status Check

No, the repository has **not** yet completed the full conference-grade experimental package that would be needed to support the intended claims.

What is already present:

- the proposal and research framing;
- a detailed experimental blueprint;
- an initial minimal reproducible prototype for RL-guided region repair.

What is **not** yet fully completed at paper-ready scale:

- **small / medium / large** benchmark sweeps with fixed compute budgets;
- a **broad baseline suite** spanning exact OR, decomposition, metaheuristics, and modern neural schedulers;
- **out-of-distribution generalization** experiments across scale, flexibility, disturbance type, and benchmark family;
- **component ablations** and **representation / reward / training** ablations;
- **public / real benchmark** evaluation such as Brandimarte, Hurink, and related FJSP/JSSP families;
- **multi-seed statistical reporting** with confidence intervals and paired significance tests;
- solver-backend cross-checks and reviewer-facing backup experiments.

In short: the current repo should be read as a **research buildout toward a top-tier paper**, not as a finished empirical package.

## Literature-Grounded Experimental Expectations

The experiment design below is aligned with the standards commonly seen in representative scheduling papers and surveys, especially:

- **L2D**: *Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning* (2020), which set the pattern of comparing against hand-crafted dispatching rules and testing generalization to larger unseen instances;
- **DAN**: *Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning* (2023), which reinforced the expectation of using both synthetic data and public FJSP benchmarks;
- **DRL-Guided Improvement Heuristic** (ICLR 2024), which raised the bar for improvement-based RL baselines and stronger local-search comparisons;
- **Survey**: *Graph Neural Networks for Job Shop Scheduling Problems: A Survey* (2024), which makes clear that modern scheduling papers are increasingly judged on benchmark breadth, fair baseline coverage, generalization, and statistical rigor.

That literature consensus is the reason this README now treats the following experimental package as the minimum scope for a credible conference submission.

## Proposed Method

The proposal studies an **unrolled LBBD solver**:

1. The **master problem** chooses high-level repair decisions such as machine assignment, freeze/unfreeze decisions, and repair region candidates.
2. The **subproblem** performs exact or near-exact local scheduling under the current repair region.
3. The solver adds **feasibility, critical-path, optimality, and stability-aware cuts**.
4. An RL policy observes the current solver state and decides:
   - which region to repair,
   - how large the repair neighborhood should be,
   - how much subproblem budget to spend,
   - and, in later versions, which cuts to prioritize.

The key point is that learning controls the **search process**, not the full schedule directly.

## Experimental Plan

The section below is written as a paper-grade experimental blueprint rather than a lightweight project note. It is intended to support the core experimental section of a top-tier ML/optimization paper.

### 1. Experimental Objectives and Core Claims

The experiments should validate five central claims.

1. **Anytime superiority**: RL-guided LBBD produces better repair solutions than plain LBBD and hand-designed repair heuristics under short and medium wall-clock budgets.
2. **Structured learning advantage**: controlling a decomposition solver is better than directly generating schedules with pure neural dispatching, especially in feasibility recovery and stability preservation.
3. **Generalization**: the learned controller transfers across instance sizes, flexibility levels, disturbance mixtures, and benchmark families not seen during training.
4. **Robustness without losing exactness**: the learning component improves search order under limited budget while fallback LBBD preserves completeness in the long-budget regime.
5. **Attribution**: gains come from specific components of the proposed method, especially region selection, budget control, cut prioritization, imitation pretraining, and heterogeneous graph encoding.

To satisfy reviewer expectations, the experimental narrative should be organized around these claims rather than around the implementation order.

### 1.5. Minimum Experimental Package for a Top-Tier Submission

If this work is to support a serious NeurIPS / ICLR / ICML / Operations Research style submission, the following blocks should all exist in some form.

| Block | What must be run | Why reviewers will ask for it |
|---|---|---|
| Size ladder | small / medium / large instances under matched budgets | verifies scaling instead of cherry-picking one regime |
| Broad baselines | exact, LBBD, heuristic, metaheuristic, neural | prevents the comparison set from looking narrow |
| Public benchmarks | Brandimarte, Hurink, and related families | shows contact with established literature |
| Dynamic benchmarks | converted repair tasks from static schedules + synthetic dynamic suites | demonstrates relevance to rescheduling rather than only static scheduling |
| Generalization | ID, scale-OOD, flexibility-OOD, disturbance-OOD, family-OOD | supports claims beyond memorizing one generator |
| Ablations | component, representation, reward, training protocol | proves where gains come from |
| Statistical rigor | at least 5 seeds, CIs, paired tests, effect sizes | prevents single-run overclaiming |
| Reviewer backups | equal-compute checks, long-budget exactness, solver-portability subset | de-risks predictable rebuttal attacks |

Anything materially short of this matrix should be described as a **prototype** or **partial validation**, not as final paper evidence.

### 2. Datasets and Preprocessing

We recommend evaluating on three complementary groups of instances.

#### A. Public static benchmarks converted into dynamic repair tasks

Use benchmark families widely adopted in JSSP/FJSP research:

- **Brandimarte (MK)** for standard FJSP evaluation;
- **Hurink** families for different flexibility regimes;
- **Dauzere-Peres / Paulli** style flexible-shop instances;
- **Taillard / Lawrence / ABZ-style JSSP families** as low-flexibility or zero-flexibility stress tests.

Each static instance is first solved into an initial production plan using a strong offline solver pipeline:

1. CP-SAT / CP Optimizer with a generous time limit;
2. warm starts from strong heuristics if needed;
3. keep the best incumbent as the baseline schedule `x0`.

Then convert the static schedule into a dynamic rescheduling task by injecting one or more disturbances at a rescheduling time `t_d`.

#### B. Synthetic dynamic FJSP instances

Generate controlled synthetic instances with:

- jobs `J in {10, 20, 50, 100}`,
- machines `M in {5, 10, 20}`,
- operations per job sampled from `Uniform{5, 20}`,
- flexibility ratio `f in {0.2, 0.5, 0.8}`,
- due-date tightness factors in `{1.2, 1.5, 2.0}`,
- utilization levels spanning moderate to high congestion.

Dynamic disturbances should include:

- **machine breakdowns**: downtime windows sampled at random times with duration `5%-20%` of the nominal horizon;
- **new job arrivals**: inject `5%-30%` extra jobs after `t_d`;
- **processing-time inflation**: multiply a subset of operation times by `1.2-1.8`;
- **urgent jobs**: insert jobs with tight due dates and high tardiness penalty;
- **mixed disturbances**: combine at least two disturbance types.

#### C. Cross-distribution generalization suites

To make generalization claims credible, separate train and test distributions deliberately:

- **ID test**: same scale and same disturbance distribution as training, unseen random seeds;
- **Scale-OOD**: train on `{10, 20}` jobs and test on `{50, 100}`;
- **Flexibility-OOD**: train on medium flexibility and test on very low / very high flexibility;
- **Disturbance-OOD**: train on single disturbances and test on mixed disturbances;
- **Family-OOD**: train only on synthetic instances and evaluate zero-shot on public benchmark families.

#### Preprocessing and Split Protocol

- Split synthetic data by generator seed with an `80/10/10` train/validation/test ratio.
- Keep benchmark families completely isolated from training and hyperparameter tuning.
- Normalize processing times, due dates, and temporal features by the initial schedule horizon or instance-level statistics.
- Build the disturbed-state graph using only causal information available at rescheduling time; no future leakage is allowed.
- For each base instance, generate `K=10` disturbance realizations for test-time paired comparison.

### 3. Baseline Selection

Baselines should cover exact optimization, decomposition, metaheuristics, and modern learning methods.

#### A. Optimization and rescheduling baselines

- **Full CP-SAT Rescheduling**: recompute the disturbed schedule from scratch under the same online time budget.
- **Full CP Optimizer / MIP**: small- and medium-scale exact baseline for strong lower/upper references.
- **Rolling Horizon Repair**: classic industrial rescheduling baseline.
- **ALNS / Ruin-and-Recreate**: strong large-neighborhood search baseline.
- **Tabu Search / Simulated Annealing / GA**: representative metaheuristics.

These baselines answer the reviewer question: "Does the proposed hybrid method actually outperform serious OR competitors?"

#### B. Decomposition baselines

- **Plain LBBD**: fixed repair region, fixed cut management, fixed subproblem budget.
- **Critical-Path LBBD**: hand-designed region expansion from critical path / affected machine blocks.
- **Random-Region LBBD**: stochastic neighborhood baseline.
- **Oracle-Region LBBD**: offline upper-bound baseline that picks the best region among candidates with hindsight.

These baselines isolate the value of learning inside the same algorithmic skeleton.

#### C. Learning baselines

- **Dispatching Rules**: SPT, LPT, MWKR, EDD, CR and related rules.
- **L2D-style GNN dispatching**: construction-based RL for JSSP.
- **DAN-style FJSP RL**: end-to-end RL with operation-machine attention.
- **DRL-guided improvement heuristic**: improvement-based RL rather than one-shot dispatching.
- **CP-guided RL dispatching**: RL with a CP backend for schedule construction or refinement.
- **Dynamic RL rescheduling baseline**: a prior DRL method designed for online arrivals or dynamic shop-floor events.

#### Fairness Rules for Baselines

- All methods receive the **same online wall-clock budget** per instance.
- All learning baselines receive the **same training-step budget** and the same validation-based early-stopping protocol.
- All non-learning baselines receive the **same hyperparameter tuning budget** on the validation split.
- If a baseline does not natively support stability-aware repair, add the same stability-regularized objective whenever possible and state the adaptation explicitly.

### 4. Evaluation Metrics

#### Primary metrics

1. **Normalized repair objective** `Obj_norm`:
   the weighted objective from the proposal, normalized per instance so results are comparable across scales.
2. **Anytime AUC**:
   area under the objective-time curve over budgets `{1s, 5s, 10s, 30s, 60s, 300s}`.
3. **Feasibility rate**:
   percentage of disturbed instances for which a feasible repair is found within budget.

These are the three headline metrics and should appear in the main tables.

#### Secondary metrics

- **Makespan** `C_max`;
- **total tardiness**;
- **average start-time deviation**;
- **machine reassignment rate**;
- **sequence perturbation distance**;
- **time to first feasible solution**;
- **final optimality gap** to the best-known solution or strongest exact reference;
- **number of LBBD iterations**;
- **number of cuts generated / accepted**;
- **subproblem solve time** and **master solve time**.

#### Statistical reporting

- For every learned method, report **mean ± standard deviation** across `5` independent training seeds.
- For every test setting, aggregate over all instance-disturbance pairs and additionally report **95% confidence intervals**.
- Use **paired Wilcoxon signed-rank tests** or **paired bootstrap tests** with Holm-Bonferroni correction for multiple comparisons.
- Report an effect size such as **Cliff's delta** or paired standardized mean difference.

### 5. Experimental Setup and Hyperparameters

#### Software stack

- PyTorch `2.x`
- CUDA `12.x`
- OR-Tools CP-SAT `9.x`
- IBM CP Optimizer `22.x` if available
- Gurobi `11.x` for MIP references where appropriate

#### RL training protocol

- algorithm: **PPO**
- optimizer: **AdamW**
- learning rate: `{1e-4, 3e-4}` search grid; default `3e-4`
- value loss coefficient: `0.5`
- entropy coefficient: `{0.001, 0.01}`
- PPO clip ratio: `0.2`
- discount factor: `0.99`
- GAE lambda: `0.95`
- gradient clipping: `1.0`
- rollout horizon: `256` decision steps
- PPO epochs per update: `4`
- mini-batch size: `64`
- batch size in environments: `32-64` instances

#### Model configuration

- heterogeneous GNN encoder with `4` message-passing layers
- hidden dimension `{128, 256}`
- dropout `{0.0, 0.1}`
- separate policy heads for anchor, type, radius, and budget
- optional cut-ranking head activated only in the full model

#### Imitation pretraining

- expert pool: critical-path-first, affected-machine-first, largest-delay-first, best-improvement LBBD oracle on small instances
- pretraining samples: at least `100k` state-action pairs
- epochs: `20-50`
- validation by held-out trajectories

#### Compute and reproducibility

- training on `4-8` GPUs of A100 class or equivalent
- evaluation on fixed CPU machines with a specified thread budget, e.g. `16` threads per solver
- each method tested with the same wall-clock limits
- fix and publish all random seeds
- publish benchmark generation code, disturbance seeds, solver parameter files, and evaluation scripts

#### Tuning budget

- learning methods: same maximum training steps or same GPU-hour budget
- search/metaheuristic baselines: same number of validation trials, e.g. `200` Optuna trials or an equal CPU-hour envelope

This section is essential for preempting fairness complaints.

### 6. Main Results and Expected Analysis

The main paper should include at least the following tables.

#### Table 1. In-distribution dynamic FJSP performance

| Method | Obj_norm ↓ | Anytime AUC ↓ | Feasibility ↑ | Stability cost ↓ | Time-to-feasible ↓ |
|---|---:|---:|---:|---:|---:|
| Full CP-SAT |  |  |  |  |  |
| Plain LBBD |  |  |  |  |  |
| Critical-Path LBBD |  |  |  |  |  |
| ALNS |  |  |  |  |  |
| DAN-style RL |  |  |  |  |  |
| Proposed RL-LBBD |  |  |  |  |  |

#### Table 2. Out-of-distribution generalization

| Method | Scale-OOD Obj ↓ | Disturbance-OOD Obj ↓ | Flexibility-OOD Obj ↓ | Family-OOD Obj ↓ |
|---|---:|---:|---:|---:|
| Plain LBBD |  |  |  |  |
| DAN-style RL |  |  |  |  |
| Improvement RL |  |  |  |  |
| Proposed RL-LBBD |  |  |  |  |

#### Table 3. Disturbance-type breakdown

| Method | Breakdown ↓ | New arrivals ↓ | Proc.-time shift ↓ | Urgent jobs ↓ | Mixed ↓ |
|---|---:|---:|---:|---:|---:|

#### Table 4. Efficiency and compute profile

| Method | Online time budget | Mean runtime | GPU use | Training cost | Test cost / instance |
|---|---:|---:|---:|---:|---:|

#### Expected reading of the results

- **Full CP-SAT / CP Optimizer** may remain strongest on very small instances with long budgets.
- **Pure RL dispatching** should be faster but weaker in feasibility and stability.
- **Plain LBBD** should be competitive but less sample-efficient in the search order.
- **Proposed RL-LBBD** is expected to win primarily on anytime AUC, feasibility under tight budgets, and stability-aware repair quality, especially on medium/large and mixed-disturbance cases.

The paper should avoid overselling exact-optimality superiority and instead emphasize the correct claim: **better search control under practical time budgets**.

### 7. Ablation Design

The ablation section should be incremental and attributable.

#### A. Component ablations

- `A0`: Plain LBBD
- `A1`: `A0 +` learned region selection
- `A2`: `A1 +` learned budget control
- `A3`: `A2 +` learned cut prioritization
- `A4`: `A3 +` imitation pretraining
- `A5`: `A4 +` heterogeneous graph encoder
- `A6`: `A5 +` stability-aware features
- `A7`: `A6 +` gap-aware reward
- `A8`: `A7 +` fallback exact mode

#### B. Representation ablations

- homogeneous GNN vs. heterogeneous GNN
- no cut nodes
- no disturbance nodes
- no historical trajectory features
- no critical-path indicator

#### C. Learning ablations

- PPO vs. imitation-only
- with vs. without KL regularization to expert policy
- dense reward vs. sparse terminal reward
- fixed vs. adaptive subproblem budget action space

Each ablation should report both quality and feasibility; reporting objective alone is not enough.

### 8. Deeper Analysis

To survive strong reviews, include analyses beyond the standard tables.

#### A. Sensitivity analysis

- objective weights `(alpha, beta, gamma, eta, kappa)`
- repair radius range
- cut-pool size
- maximum LBBD iterations
- subproblem time budget

#### B. Scaling analysis

Plot performance against:

- number of jobs,
- number of machines,
- flexibility ratio,
- disturbance severity.

#### C. Training dynamics

- reward vs. environment steps
- validation objective vs. training epoch
- imitation pretraining warm-start benefit
- variance across seeds

#### D. Visual and qualitative analysis

- visualize selected repair regions
- visualize cut-pool evolution
- compare repair trajectories against critical-path heuristics
- show representative success and failure cases

#### E. Efficiency analysis

- master vs. subproblem runtime decomposition
- performance profiles / Dolan-More curves
- memory usage
- GPU inference overhead at test time

#### F. Statistical robustness

- paired significance tests against plain LBBD and best non-learning baseline
- confidence bands on anytime curves
- win/tie/loss counts across instances

### 9. Reviewer-Facing Backup Experiments

This section should be planned before submission, not after rejection.

#### Likely attack 1: "The gains come from extra compute, not from the method."

**Defense**:

- equal online wall-clock budgets for all methods;
- equal hyperparameter tuning budget;
- report offline training cost separately;
- compare against imitation-only and oracle-region baselines.

#### Likely attack 2: "The policy just learns a critical-path heuristic."

**Defense**:

- measure overlap between selected repair regions and critical-path regions;
- remove critical-path features and observe degradation;
- compare against a strong critical-path LBBD tuned with the same budget.

#### Likely attack 3: "The learned controller breaks LBBD exactness."

**Defense**:

- run long-budget experiments on small instances;
- show convergence to the same optimum / best-known solution as plain LBBD when fallback mode is enabled;
- report no-regression feasibility results.

#### Likely attack 4: "The method is overfit to synthetic disturbances."

**Defense**:

- zero-shot evaluation on public benchmark families;
- leave-one-disturbance-out training;
- severity-shift stress tests;
- if possible, a small semi-realistic simulator with correlated disturbance processes.

#### Likely attack 5: "Stability gains are due to objective weighting rather than better search."

**Defense**:

- plot Pareto fronts between efficiency and stability;
- compare methods at matched stability levels;
- report hypervolume over multi-objective trade-offs.

#### Likely attack 6: "The approach is too solver-specific."

**Defense**:

- re-run a subset with both CP-SAT and CP Optimizer backends;
- show that ranking trends are preserved.

#### Likely attack 7: "Variance is too high."

**Defense**:

- five training seeds minimum;
- confidence intervals;
- significance testing;
- win-rate tables across instance families.

### 10. Limitations of the Experimental Scope

Even a strong paper should state its empirical limits clearly.

- The dynamic environment is still partly synthetic and may not capture all MES-level shop-floor constraints.
- Training cost may be high because each episode calls exact or near-exact subproblems.
- Some strong OR baselines may require substantial engineering, which limits the breadth of reproducible comparisons.
- Stability is represented by weighted penalties; truly interactive human-planner preferences are not modeled.
- The study may not fully cover stochastic durations, energy objectives, maintenance coupling, or multi-line coordination.
- Results on public benchmarks do not automatically imply deployment readiness in a real factory.

## Immediate Next Steps

Recommended implementation order:

1. build the disturbance generator and metric suite;
2. implement a reproducible plain LBBD baseline;
3. add public benchmark loaders for Brandimarte / Hurink style families;
4. define the state, action, and reward interface for RL-Region-LBBD;
5. expand the baseline suite to CP-SAT, ALNS, dispatching, and neural comparators;
6. collect expert trajectories for imitation pretraining;
7. add PPO fine-tuning, multi-seed evaluation, and anytime curves;
8. run the full generalization and ablation matrix before making strong paper claims.

## README Scope Note

This README now serves two purposes at once:

- it documents the current repository honestly;
- it defines the **experiment package that still needs to be executed** for the project to be submission-ready.

Readers should therefore distinguish carefully between:

- **what has been implemented or minimally run already**, and
- **what is required for a conference-grade empirical section**.

## Citation / Source

This README is derived from the proposal in:

- `lbbd_rl_unroll_proposal.tex`
- `lbbd_rl_unroll_proposal.pdf`

## License

No license has been added yet.
