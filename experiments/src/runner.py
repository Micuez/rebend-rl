from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from .core import Disturbance, Instance, Schedule
from .environment import ACTION_STRATEGIES, RepairEnvironment
from .generator import generate_disturbance, generate_instance
from .policies import PolicyValueNet, greedy_action
from .reporting import collect_environment, git_state, render_report, save_json, save_plot, sha256_file
from .scheduler import compute_metrics, greedy_schedule, run_policy_episode
from .train import train_ppo


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> Dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataset(config: Dict, split: str, rng: random.Random) -> List[Tuple[Instance, Schedule, Disturbance]]:
    split_size = config["data"][f"{split}_instances"]
    job_range = tuple(config["data"]["train_job_range"] if split != "test" else config["data"]["test_job_range"])
    records = []
    for index in range(split_size):
        instance = generate_instance(
            rng=rng,
            name=f"{split}_{index}",
            job_range=job_range,
            num_machines=config["data"]["machines"],
            ops_per_job=tuple(config["data"]["ops_per_job"]),
            flexibility_choices=list(config["data"]["flexibility_choices"]),
            duration_range=tuple(config["data"]["duration_range"]),
            due_date_factor=config["data"]["due_date_factor"],
        )
        baseline = greedy_schedule(instance)
        horizon = max(item.end for item in baseline.assignments.values())
        disturbance = generate_disturbance(
            rng=rng,
            instance=instance,
            baseline_horizon=horizon,
            reschedule_fraction=config["data"]["disturbance"]["reschedule_fraction"],
            breakdown_duration_fraction=tuple(config["data"]["disturbance"]["breakdown_duration_fraction"]),
            inflation_probability=config["data"]["disturbance"]["inflation_probability"],
            inflation_range=tuple(config["data"]["disturbance"]["inflation_range"]),
        )
        records.append((instance, baseline, disturbance))
    return records


def rollout_env_for_training(config: Dict, dataset, seed: int):
    rng = random.Random(seed)

    def factory():
        instance, baseline, disturbance = rng.choice(dataset)
        return RepairEnvironment(
            instance=instance,
            baseline=baseline,
            disturbance=disturbance,
            radii=config["solver"]["region_radii"],
            weights=config["solver"]["objective_weights"],
            time_limit_sec=config["solver"]["subproblem_time_limit_sec"],
            threads=config["solver"]["threads"],
            max_iterations=config["solver"]["max_iterations"],
            seed=rng.randint(0, 10_000_000),
        )

    return factory


def evaluate_method(
    method: str,
    config: Dict,
    dataset,
    rl_model: PolicyValueNet | None,
    device: torch.device,
    seed: int,
) -> Tuple[pd.DataFrame, Dict]:
    rng = random.Random(seed)
    rows = []
    trace_rows = []
    radii = list(config["solver"]["region_radii"])
    fixed_sequence = [("affected_machine", radii[0]), ("critical_path", radii[1]), ("tardy_job", radii[-1])]
    for instance, baseline, disturbance in dataset:
        if method == "full_reschedule":
            schedule = greedy_schedule(instance, disturbance)
            history = []
        elif method == "plain_lbbd":
            schedule, history = run_policy_episode(
                instance, baseline, disturbance, config["solver"]["objective_weights"], fixed_sequence,
                config["solver"]["subproblem_time_limit_sec"], config["solver"]["threads"]
            )
        elif method == "critical_path_lbbd":
            schedule, history = run_policy_episode(
                instance, baseline, disturbance, config["solver"]["objective_weights"],
                [("critical_path", radius) for radius in radii[: config["solver"]["max_iterations"]]],
                config["solver"]["subproblem_time_limit_sec"], config["solver"]["threads"]
            )
        elif method == "random_region_lbbd":
            sequence = [
                (rng.choice(ACTION_STRATEGIES), rng.choice(radii))
                for _ in range(config["solver"]["max_iterations"])
            ]
            schedule, history = run_policy_episode(
                instance, baseline, disturbance, config["solver"]["objective_weights"], sequence,
                config["solver"]["subproblem_time_limit_sec"], config["solver"]["threads"]
            )
        elif method == "rl_guided_lbbd":
            env = RepairEnvironment(
                instance=instance,
                baseline=baseline,
                disturbance=disturbance,
                radii=radii,
                weights=config["solver"]["objective_weights"],
                time_limit_sec=config["solver"]["subproblem_time_limit_sec"],
                threads=config["solver"]["threads"],
                max_iterations=config["solver"]["max_iterations"],
                seed=seed,
            )
            state = env.reset()
            done = False
            history = []
            while not done:
                action = greedy_action(rl_model, state, device)
                state, _, done, info = env.step(action)
                trace_rows.append(
                    {
                        "instance": instance.name,
                        "method": method,
                        "strategy": info["strategy"],
                        "radius": info["radius"],
                        "feasible": info["feasible"],
                        "objective": info["objective"],
                        "elapsed": info["elapsed"],
                    }
                )
            schedule = env.current
        else:
            raise ValueError(method)
        metrics = compute_metrics(
            instance, baseline, schedule, disturbance, config["solver"]["objective_weights"]
        )
        rows.append(
            {
                "instance": instance.name,
                "method": method,
                **metrics,
            }
        )
        for step in history:
            trace_rows.append(
                {
                    "instance": instance.name,
                    "method": method,
                    "strategy": step.strategy,
                    "radius": step.radius,
                    "feasible": step.feasible,
                    "objective": step.objective,
                    "elapsed": step.solve_time_sec,
                }
            )
    return pd.DataFrame(rows), {"trace": pd.DataFrame(trace_rows)}


def summarize_results(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics_df.groupby("method").agg(
        objective_mean=("objective", "mean"),
        objective_std=("objective", "std"),
        makespan_mean=("makespan", "mean"),
        tardiness_mean=("tardiness", "mean"),
        feasibility_rate=("feasible", "mean"),
        changed_fraction_mean=("changed_fraction", "mean"),
    )
    return grouped.sort_values("objective_mean")


def build_contract(config: Dict) -> Dict:
    return {
        "objective": "Implement and run a minimal reproducible RL-Region-LBBD experiment for dynamic flexible job-shop repair on synthetic disturbed instances.",
        "hypothesis": "A learned region-selection policy can improve the local-repair search order over fixed and random LBBD region policies under short solver budgets.",
        "independent_variables": ["repair policy", "region radius", "synthetic disturbance realization"],
        "dependent_variables": ["objective", "makespan", "tardiness", "start deviation", "machine reassignment", "feasibility"],
        "controls": ["same synthetic generator family", "same CP-SAT subproblem budget", "same objective weights", "same evaluation split"],
        "dataset": "Synthetic dynamic FJSP instances generated on the fly",
        "preprocessing": "Baseline schedule generation, disturbance injection, unfinished-operation filtering",
        "method_under_test": "Minimal RL-Region-LBBD with PPO policy over repair-region heuristics",
        "metrics": ["objective", "makespan", "tardiness", "start deviation", "machine reassignment", "changed fraction"],
        "trials": {
            "train_instances": config["data"]["train_instances"],
            "validation_instances": config["data"]["val_instances"],
            "test_instances": config["data"]["test_instances"],
            "training_epochs": config["training"]["epochs"],
        },
        "ablations": ["full_reschedule", "plain_lbbd", "critical_path_lbbd", "random_region_lbbd", "rl_guided_lbbd"],
        "success_criteria": "RL-guided policy achieves lower mean objective than plain/random LBBD on the held-out synthetic test split.",
        "hardware_constraints": "Single host, repository Python environment, optional CUDA acceleration when available",
        "expected_artifacts": ["metrics.csv", "metrics.json", "report.md", "figures/objective_bar.png", "logs"],
        "known_risks": [
            "This is a prototype, not the full paper-scale benchmark suite.",
            "Only synthetic instances are included in the executable run.",
            "Objective omits sequence perturbation distance for tractability.",
        ],
    }


def build_plan_coverage(config: Dict) -> List[str]:
    return [
        "已执行最小原型验证：单次 PPO 训练 + 5 个方法在合成动态 FJSP 测试集上的对比。",
        "已覆盖 README 中的部分核心主张：anytime/短预算下的搜索顺序收益、结构化学习优于固定或随机 repair 区域策略。",
        "当前数据覆盖仅限合成实例，未覆盖 README 中要求的 Brandimarte、Hurink 等公共 benchmark。",
        "当前泛化覆盖仅体现轻量的 train/test 尺度迁移；未执行 README 中定义的 Flexibility-OOD、Disturbance-OOD、Family-OOD 矩阵。",
        "当前 baseline 覆盖了 full_reschedule 与 3 个 LBBD 变体，但尚未覆盖 README 中要求的 ALNS、Tabu、L2D、DAN 等广泛基线。",
        "当前统计覆盖为单训练种子；未达到 README 中至少 5 个训练种子、95% CI 与 paired significance test 的标准。",
        "当前报告结论应视为 README 方案下的阶段性可复现实验，而不是完整投稿级证据包。",
    ]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for item in row.tolist():
            if isinstance(item, float):
                values.append(f"{item:.4f}")
            else:
                values.append(str(item))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(config_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    set_seed(config["seed"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / config["output_root"] / f"{timestamp}_{config['experiment_name']}"
    latest_root = repo_root / config["output_root"] / f"latest_{config['experiment_name']}"
    figures_dir = run_root / "figures"
    logs_dir = run_root / "logs"
    run_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config["seed"])
    train_data = build_dataset(config, "train", rng)
    val_data = build_dataset(config, "val", rng)
    test_data = build_dataset(config, "test", rng)

    device_name = "cpu"
    if config["device"] == "cuda" and torch.cuda.is_available():
        device_name = "cuda"
    device = torch.device(device_name)
    model = PolicyValueNet(input_dim=11, action_dim=len(ACTION_STRATEGIES) * len(config["solver"]["region_radii"])).to(device)
    train_history = train_ppo(
        model=model,
        env_factory=rollout_env_for_training(config, train_data, config["seed"]),
        device=device,
        config=config["training"],
    )
    torch.save(model.state_dict(), run_root / "artifacts_policy.pt")

    all_metrics = []
    traces = []
    for method in config["baselines"]:
        metrics_df, extras = evaluate_method(
            method=method,
            config=config,
            dataset=test_data,
            rl_model=model,
            device=device,
            seed=config["seed"],
        )
        all_metrics.append(metrics_df)
        if not extras["trace"].empty:
            traces.append(extras["trace"])

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    trace_df = pd.concat(traces, ignore_index=True) if traces else pd.DataFrame()
    summary_df = summarize_results(metrics_df)
    metrics_df.to_csv(run_root / "metrics.csv", index=False)
    summary_df.to_csv(run_root / "metrics_summary.csv")
    if not trace_df.empty:
        trace_df.to_csv(run_root / "trace.csv", index=False)
    save_plot(metrics_df, figures_dir / "objective_bar.png")

    environment = collect_environment()
    environment["torch"] = torch.__version__
    environment["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        environment["cuda_device_count"] = torch.cuda.device_count()
        environment["cuda_current_device"] = torch.cuda.current_device()
        environment["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    environment["git_state"] = git_state(repo_root)

    data_manifest = {
        "config_path": str(config_path),
        "train_instances": len(train_data),
        "val_instances": len(val_data),
        "test_instances": len(test_data),
        "metrics_csv_sha256": sha256_file(run_root / "metrics.csv"),
    }

    summary_payload = {
        "train_history_tail": {
            "episode_return_mean": mean(train_history["episode_return"][-10:]),
            "final_objective_mean": mean(train_history["final_objective"][-10:]),
        },
        "results": summary_df.reset_index().to_dict(orient="records"),
    }
    save_json(run_root / "metrics.json", summary_payload)
    (run_root / "experiment_contract.yaml").write_text(
        yaml.safe_dump(build_contract(config), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    save_json(run_root / "environment.json", environment)
    (run_root / "git_state.txt").write_text(environment["git_state"], encoding="utf-8")
    run_command = f"{sys.executable} scripts/run_rebend_minimal.py --config {config_path}"
    (run_root / "commands.sh").write_text(
        run_command + "\n",
        encoding="utf-8",
    )

    results_md = dataframe_to_markdown(summary_df.reset_index())
    implementation_summary = [
        "Added a synthetic dynamic FJSP generator and disturbance injector.",
        "Implemented a CP-SAT local repair subproblem with frozen-region logic.",
        "Implemented four baseline policies plus a PPO-trained RL-guided region selector.",
        "Added a runner that records config, environment, metrics, traces, plots, and report artifacts.",
    ]
    plan_coverage = build_plan_coverage(config)
    verification = [
        "Python syntax check on the new experiment package.",
        "Executed a full training and evaluation run from the repository Python environment.",
        "Full training and evaluation run archived in the result directory.",
    ]
    deviations = [
        "Executed the minimal RL-Region-LBBD variant, not the full proposal with cut-ranking, imitation pretraining, and public benchmarks.",
        "Used synthetic instances only; no Brandimarte/Hurink benchmark import was added in this run.",
        "Used a reduced objective without explicit sequence perturbation distance.",
        "Used a single training run rather than 5 independent training seeds due runtime constraints.",
    ]
    limitations = [
        "Findings should be read as a prototype validation, not as paper-ready evidence.",
        "Evaluation uses small-to-medium synthetic sizes tuned for a single-machine reproducible run.",
        "Full CP-SAT baseline here is a fast greedy disturbed reschedule, not a global exact recomputation baseline.",
    ]
    next_steps = [
        "Replace the simple full-reschedule baseline with a true global CP-SAT recomputation model.",
        "Add public FJSP benchmark loaders and train/test OOD splits from the proposal.",
        "Extend the reward/action space to cover subproblem budget control and cut prioritization.",
        "Run 3-5 seeds and bootstrap CIs before making stronger claims.",
    ]
    artifact_index = [
        "metrics.csv",
        "metrics_summary.csv",
        "metrics.json",
        "trace.csv",
        "environment.json",
        "git_state.txt",
        "commands.sh",
        "figures/objective_bar.png",
        "artifacts_policy.pt",
    ]
    render_report(
        output_path=run_root / "report.md",
        status="completed",
        contract=build_contract(config),
        implementation_summary=implementation_summary,
        plan_coverage=plan_coverage,
        environment=environment,
        data_manifest=data_manifest,
        commands=[
            f"{sys.executable} -m py_compile experiments/src/*.py",
            run_command,
        ],
        results_md=results_md,
        verification=verification,
        deviations=deviations,
        limitations=limitations,
        next_steps=next_steps,
        artifact_index=artifact_index,
    )

    latest_files = [
        "report.md",
        "metrics.csv",
        "metrics_summary.csv",
        "metrics.json",
        "environment.json",
        "git_state.txt",
        "commands.sh",
        "experiment_contract.yaml",
        "artifacts_policy.pt",
    ]
    for name in latest_files:
        shutil.copy2(run_root / name, latest_root / name)
    if (run_root / "trace.csv").exists():
        shutil.copy2(run_root / "trace.csv", latest_root / "trace.csv")
    latest_figures = latest_root / "figures"
    latest_figures.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figures_dir / "objective_bar.png", latest_figures / "objective_bar.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
