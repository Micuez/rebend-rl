from __future__ import annotations

import copy
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from .environment import ACTION_STRATEGIES, RepairEnvironment
from .policies import PolicyValueNet, greedy_action
from .reporting import collect_environment, git_state, save_json, sha256_file
from .runner import build_dataset, load_config, set_seed
from .scheduler import compute_metrics, greedy_schedule, run_policy_episode
from .train import train_ppo


@dataclass
class TrainedPolicy:
    name: str
    family: str
    model: PolicyValueNet
    device: torch.device
    action_strategies: Sequence[str]
    radii: Sequence[int]
    reward_config: Dict[str, float]
    training_summary: Dict[str, object]


def merge_dict(base: Dict, override: Dict) -> Dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_env(instance, baseline, disturbance, solver_cfg: Dict, seed: int, action_strategies=None, reward_config=None):
    return RepairEnvironment(
        instance=instance,
        baseline=baseline,
        disturbance=disturbance,
        radii=solver_cfg["region_radii"],
        weights=solver_cfg["objective_weights"],
        time_limit_sec=solver_cfg["subproblem_time_limit_sec"],
        threads=solver_cfg["threads"],
        max_iterations=solver_cfg["max_iterations"],
        seed=seed,
        action_strategies=action_strategies,
        reward_config=reward_config,
    )


def rollout_policy_model(policy: TrainedPolicy, solver_cfg: Dict, dataset, seed: int, scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    traces: List[Dict] = []
    for index, (instance, baseline, disturbance) in enumerate(dataset):
        env = make_env(
            instance=instance,
            baseline=baseline,
            disturbance=disturbance,
            solver_cfg=merge_dict(solver_cfg, {"region_radii": list(policy.radii)}),
            seed=seed * 10_000 + index,
            action_strategies=policy.action_strategies,
            reward_config=policy.reward_config,
        )
        state = env.reset()
        done = False
        while not done:
            action = greedy_action(policy.model, state, policy.device)
            state, reward, done, info = env.step(action)
            traces.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "instance": instance.name,
                    "method": policy.name,
                    "family": policy.family,
                    "iteration": env.iteration,
                    "strategy": info["strategy"],
                    "radius": info["radius"],
                    "feasible": info["feasible"],
                    "accepted": info["accepted"],
                    "objective": info["objective"],
                    "reward": reward,
                    "elapsed": info["elapsed"],
                }
            )
        metrics = compute_metrics(instance, baseline, env.current, disturbance, solver_cfg["objective_weights"])
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "instance": instance.name,
                "method": policy.name,
                "family": policy.family,
                **metrics,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(traces)


def oracle_action(env: RepairEnvironment) -> int:
    best_index = 0
    best_key = None
    for action_index in range(len(env.action_space)):
        preview = env.preview_action(action_index)
        key = (
            preview["objective"],
            0 if preview["feasible"] else 1,
            preview["elapsed"],
            action_index,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_index = action_index
    return best_index


def rollout_oracle(solver_cfg: Dict, dataset, seed: int, scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    traces: List[Dict] = []
    for index, (instance, baseline, disturbance) in enumerate(dataset):
        env = make_env(instance, baseline, disturbance, solver_cfg, seed * 10_000 + index)
        state = env.reset()
        done = False
        while not done:
            action = oracle_action(env)
            state, reward, done, info = env.step(action)
            traces.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "instance": instance.name,
                    "method": "oracle_region_lbbd",
                    "family": "decomposition",
                    "iteration": env.iteration,
                    "strategy": info["strategy"],
                    "radius": info["radius"],
                    "feasible": info["feasible"],
                    "accepted": info["accepted"],
                    "objective": info["objective"],
                    "reward": reward,
                    "elapsed": info["elapsed"],
                }
            )
        metrics = compute_metrics(instance, baseline, env.current, disturbance, solver_cfg["objective_weights"])
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "instance": instance.name,
                "method": "oracle_region_lbbd",
                "family": "decomposition",
                **metrics,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(traces)


def evaluate_fixed_baseline(method: str, solver_cfg: Dict, dataset, seed: int, scenario: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict] = []
    traces: List[Dict] = []
    radii = list(solver_cfg["region_radii"])
    fixed_sequence = [("affected_machine", radii[0]), ("critical_path", radii[min(1, len(radii) - 1)]), ("tardy_job", radii[-1])]
    rng = random.Random(seed)
    for instance, baseline, disturbance in dataset:
        if method == "full_reschedule":
            schedule = greedy_schedule(instance, disturbance)
            history = []
        elif method == "plain_lbbd":
            schedule, history = run_policy_episode(
                instance, baseline, disturbance, solver_cfg["objective_weights"], fixed_sequence,
                solver_cfg["subproblem_time_limit_sec"], solver_cfg["threads"]
            )
        elif method == "critical_path_lbbd":
            schedule, history = run_policy_episode(
                instance, baseline, disturbance, solver_cfg["objective_weights"],
                [("critical_path", radius) for radius in radii[: solver_cfg["max_iterations"]]],
                solver_cfg["subproblem_time_limit_sec"], solver_cfg["threads"]
            )
        elif method == "random_region_lbbd":
            sequence = [
                (rng.choice(ACTION_STRATEGIES), rng.choice(radii))
                for _ in range(solver_cfg["max_iterations"])
            ]
            schedule, history = run_policy_episode(
                instance, baseline, disturbance, solver_cfg["objective_weights"], sequence,
                solver_cfg["subproblem_time_limit_sec"], solver_cfg["threads"]
            )
        else:
            raise ValueError(method)

        metrics = compute_metrics(instance, baseline, schedule, disturbance, solver_cfg["objective_weights"])
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "instance": instance.name,
                "method": method,
                "family": "decomposition" if method != "full_reschedule" else "reference",
                **metrics,
            }
        )
        for step in history:
            traces.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "instance": instance.name,
                    "method": method,
                    "family": "decomposition" if method != "full_reschedule" else "reference",
                    "iteration": step.iteration + 1,
                    "strategy": step.strategy,
                    "radius": step.radius,
                    "feasible": step.feasible,
                    "accepted": True,
                    "objective": step.objective,
                    "reward": None,
                    "elapsed": step.solve_time_sec,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(traces)


def collect_bc_examples(solver_cfg: Dict, dataset, seed: int, max_examples: int) -> Tuple[np.ndarray, np.ndarray]:
    states: List[np.ndarray] = []
    labels: List[int] = []
    for index, (instance, baseline, disturbance) in enumerate(dataset):
        env = make_env(instance, baseline, disturbance, solver_cfg, seed * 10_000 + index)
        state = env.reset()
        done = False
        while not done and len(states) < max_examples:
            label = oracle_action(env)
            states.append(state.copy())
            labels.append(label)
            state, _, done, _ = env.step(label)
        if len(states) >= max_examples:
            break
    return np.array(states, dtype=np.float32), np.array(labels, dtype=np.int64)


def train_behavior_cloning(model: PolicyValueNet, train_states: np.ndarray, train_labels: np.ndarray, val_states: np.ndarray, val_labels: np.ndarray, cfg: Dict, device: torch.device) -> Dict[str, object]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    train_x = torch.tensor(train_states, dtype=torch.float32, device=device)
    train_y = torch.tensor(train_labels, dtype=torch.int64, device=device)
    val_x = torch.tensor(val_states, dtype=torch.float32, device=device)
    val_y = torch.tensor(val_labels, dtype=torch.int64, device=device)
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    history = {"train_loss": [], "val_accuracy": []}
    batch_size = cfg["batch_size"]

    for _ in range(cfg["epochs"]):
        indices = torch.randperm(train_x.shape[0], device=device)
        losses = []
        for start in range(0, train_x.shape[0], batch_size):
            mb = indices[start : start + batch_size]
            logits, _ = model(train_x[mb])
            loss = criterion(logits, train_y[mb])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        with torch.no_grad():
            logits, _ = model(val_x)
            preds = torch.argmax(logits, dim=-1)
            accuracy = float((preds == val_y).float().mean().item())
        history["train_loss"].append(mean(losses))
        history["val_accuracy"].append(accuracy)
        if accuracy >= best_val:
            best_val = accuracy
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {
        "train_examples": int(train_x.shape[0]),
        "val_examples": int(val_x.shape[0]),
        "best_val_accuracy": best_val,
        "history_tail": {
            "train_loss": history["train_loss"][-3:],
            "val_accuracy": history["val_accuracy"][-3:],
        },
    }


def train_policy_variant(method_cfg: Dict, solver_cfg: Dict, train_data, val_data, seed: int, device: torch.device) -> TrainedPolicy:
    local_solver = merge_dict(solver_cfg, {})
    if "region_radii" in method_cfg:
        local_solver["region_radii"] = list(method_cfg["region_radii"])
    action_strategies = method_cfg.get("action_strategies", ACTION_STRATEGIES)
    reward_config = method_cfg.get("reward_config", {})
    model = PolicyValueNet(
        input_dim=11,
        action_dim=len(action_strategies) * len(local_solver["region_radii"]),
        hidden_dim=method_cfg.get("hidden_dim", 128),
    ).to(device)
    if method_cfg["type"] == "ppo":
        history = train_ppo(
            model=model,
            env_factory=rollout_env_factory(local_solver, train_data, seed, action_strategies, reward_config),
            device=device,
            config=method_cfg["training"],
        )
        summary = {
            "family": method_cfg["family"],
            "type": "ppo",
            "episode_return_mean_tail": mean(history["episode_return"][-10:]),
            "final_objective_mean_tail": mean(history["final_objective"][-10:]),
        }
    elif method_cfg["type"] == "bc":
        bc_cfg = method_cfg["training"]
        train_states, train_labels = collect_bc_examples(local_solver, train_data, seed, bc_cfg["max_examples"])
        val_states, val_labels = collect_bc_examples(local_solver, val_data, seed + 1, max(8, bc_cfg["max_examples"] // 4))
        summary = train_behavior_cloning(model, train_states, train_labels, val_states, val_labels, bc_cfg, device)
        summary["family"] = method_cfg["family"]
        summary["type"] = "bc"
    else:
        raise ValueError(method_cfg["type"])
    return TrainedPolicy(
        name=method_cfg["name"],
        family=method_cfg["family"],
        model=model,
        device=device,
        action_strategies=action_strategies,
        radii=local_solver["region_radii"],
        reward_config=reward_config,
        training_summary=summary,
    )


def rollout_env_factory(solver_cfg: Dict, dataset, seed: int, action_strategies, reward_config):
    rng = random.Random(seed)

    def factory():
        instance, baseline, disturbance = rng.choice(dataset)
        return make_env(
            instance=instance,
            baseline=baseline,
            disturbance=disturbance,
            solver_cfg=solver_cfg,
            seed=rng.randint(0, 10_000_000),
            action_strategies=action_strategies,
            reward_config=reward_config,
        )

    return factory


def bootstrap_ci(values: Sequence[float], seed: int, repeats: int = 400, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(list(values), dtype=np.float64)
    if arr.size == 0:
        return math.nan, math.nan
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    samples = []
    for _ in range(repeats):
        sample = rng.choice(arr, size=arr.size, replace=True)
        samples.append(float(sample.mean()))
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return lower, upper


def summarize(metrics_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for (scenario, method, family), group in metrics_df.groupby(["scenario", "method", "family"]):
        lower, upper = bootstrap_ci(group["objective"].tolist(), seed + len(rows))
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                "family": family,
                "objective_mean": group["objective"].mean(),
                "objective_std": group["objective"].std(ddof=0),
                "objective_ci_low": lower,
                "objective_ci_high": upper,
                "makespan_mean": group["makespan"].mean(),
                "tardiness_mean": group["tardiness"].mean(),
                "changed_fraction_mean": group["changed_fraction"].mean(),
                "feasibility_rate": group["feasible"].mean(),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "objective_mean", "method"]).reset_index(drop=True)


def pairwise_vs_reference(metrics_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    pivot = metrics_df.pivot_table(
        index=["scenario", "seed", "instance"],
        columns="method",
        values="objective",
    ).reset_index()
    rows: List[Dict] = []
    if reference not in pivot.columns:
        return pd.DataFrame(rows)
    for scenario, group in pivot.groupby("scenario"):
        ref = group[reference]
        for method in [col for col in pivot.columns if col not in {"scenario", "seed", "instance", reference}]:
            if method not in group:
                continue
            diff = ref - group[method]
            valid = diff.dropna()
            if valid.empty:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "reference": reference,
                    "comparator": method,
                    "mean_improvement": valid.mean(),
                    "win_rate": float((valid < 0).mean()),
                    "loss_rate": float((valid > 0).mean()),
                    "ties": float((valid == 0).mean()),
                    "n": int(valid.shape[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario", "mean_improvement"])


def dataframe_to_markdown(df: pd.DataFrame, float_digits: int = 4) -> str:
    if df.empty:
        return "_empty_"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for item in row.tolist():
            if isinstance(item, float):
                values.append(f"{item:.{float_digits}f}")
            else:
                values.append(str(item))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    ensure_dir(output_dir)
    for scenario, group in summary_df.groupby("scenario"):
        plot_df = group.sort_values("objective_mean")
        plt.figure(figsize=(10, 5))
        plt.bar(plot_df["method"], plot_df["objective_mean"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Mean objective")
        plt.title(f"{scenario}: objective comparison")
        plt.tight_layout()
        plt.savefig(output_dir / f"{scenario}_objective_bar.png")
        plt.close()


def build_contract(config: Dict) -> Dict:
    return {
        "objective": "Run the first-priority conference experiment slice: ID small/medium scales, decomposition baselines, a small set of learning baselines, and core ablations.",
        "hypothesis": "RL-guided region selection improves repair quality over fixed/random decomposition baselines on both small and medium ID synthetic DFJSP repair tasks.",
        "independent_variables": [
            "scenario scale",
            "repair policy family",
            "action-space ablation",
            "reward-shaping ablation",
            "training seed",
        ],
        "dependent_variables": [
            "objective",
            "makespan",
            "tardiness",
            "changed_fraction",
            "feasibility",
        ],
        "controls": [
            "same synthetic generator within each scenario",
            "same solver time budget within each scenario",
            "same objective weights within each scenario",
            "same held-out ID test split per seed",
        ],
        "dataset": "Synthetic disturbed FJSP instances generated on the fly for ID small and ID medium settings.",
        "learning_baselines": ["bc_oracle_lbbd", "rl_guided_lbbd"],
        "decomposition_baselines": ["plain_lbbd", "critical_path_lbbd", "random_region_lbbd", "oracle_region_lbbd"],
        "ablations": ["rl_no_radius_lbbd", "rl_sparse_reward_lbbd"],
        "trials": {
            "seeds": config["seeds"],
            "scenarios": [item["name"] for item in config["scenarios"]],
        },
        "success_criteria": [
            "rl_guided_lbbd beats plain_lbbd on objective_mean in both ID scenarios",
            "rl_guided_lbbd beats random_region_lbbd on objective_mean in both ID scenarios",
            "at least one core ablation underperforms rl_guided_lbbd in each scenario",
        ],
        "known_risks": [
            "Synthetic ID evaluation only; public benchmark and OOD blocks are deferred.",
            "Learning baselines are intentionally limited to keep the first-priority slice executable.",
            "The full-reschedule reference remains a greedy disturbed reschedule, not a global exact recomputation.",
        ],
    }


def evaluate_success_criteria(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for scenario in sorted(summary_df["scenario"].unique()):
        scenario_df = summary_df[summary_df["scenario"] == scenario].set_index("method")
        checks = [
            (
                "rl_vs_plain",
                "rl_guided_lbbd objective_mean < plain_lbbd objective_mean",
                {"rl_guided_lbbd", "plain_lbbd"},
                lambda df: float(df.loc["rl_guided_lbbd", "objective_mean"]) < float(df.loc["plain_lbbd", "objective_mean"]),
            ),
            (
                "rl_vs_random",
                "rl_guided_lbbd objective_mean < random_region_lbbd objective_mean",
                {"rl_guided_lbbd", "random_region_lbbd"},
                lambda df: float(df.loc["rl_guided_lbbd", "objective_mean"]) < float(df.loc["random_region_lbbd", "objective_mean"]),
            ),
            (
                "ablation_underperforms_rl",
                "at least one core ablation has objective_mean > rl_guided_lbbd objective_mean",
                {"rl_guided_lbbd", "rl_no_radius_lbbd", "rl_sparse_reward_lbbd"},
                lambda df: any(
                    float(df.loc[item, "objective_mean"]) > float(df.loc["rl_guided_lbbd", "objective_mean"])
                    for item in ["rl_no_radius_lbbd", "rl_sparse_reward_lbbd"]
                ),
            ),
        ]
        for criterion_id, description, required_methods, predicate in checks:
            if not required_methods.issubset(scenario_df.index):
                rows.append(
                    {
                        "scenario": scenario,
                        "criterion_id": criterion_id,
                        "description": description,
                        "passed": False,
                        "detail": "missing methods required for evaluation",
                    }
                )
                continue
            passed = bool(predicate(scenario_df))
            rows.append(
                {
                    "scenario": scenario,
                    "criterion_id": criterion_id,
                    "description": description,
                    "passed": passed,
                    "detail": "pass" if passed else "fail",
                }
            )
    return pd.DataFrame(rows)


def render_report(path: Path, contract: Dict, environment: Dict, summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, training_df: pd.DataFrame, verification_df: pd.DataFrame, commands: List[str], deviations: List[str], findings: List[str], next_steps: List[str], artifact_index: List[str]) -> None:
    lines = [
        "# First-Priority Experiment Report",
        "",
        "## Status",
        "completed",
        "",
        "## Experiment Contract",
        "```yaml",
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True).strip(),
        "```",
        "",
        "## Environment",
        "```json",
        json.dumps(environment, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Commands",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "## Summary Metrics",
        dataframe_to_markdown(summary_df),
        "",
        "## Pairwise Against RL",
        dataframe_to_markdown(pairwise_df),
        "",
        "## Training Summary",
        dataframe_to_markdown(training_df),
        "",
        "## Verification",
        dataframe_to_markdown(verification_df),
        "",
        "## Findings",
        *[f"- {item}" for item in findings],
        "",
        "## Deviations",
        *[f"- {item}" for item in deviations],
        "",
        "## Next Steps",
        *[f"- {item}" for item in next_steps],
        "",
        "## Artifact Index",
        *[f"- {item}" for item in artifact_index],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(config_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = repo_root / config["output_root"] / f"{timestamp}_{config['experiment_name']}"
    latest_root = repo_root / config["output_root"] / f"latest_{config['experiment_name']}"
    figures_dir = run_root / "figures"
    models_dir = run_root / "artifacts"
    ensure_dir(figures_dir)
    ensure_dir(models_dir)
    ensure_dir(latest_root)

    device_name = "cuda" if config.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    all_metrics: List[pd.DataFrame] = []
    all_traces: List[pd.DataFrame] = []
    training_rows: List[Dict] = []

    for scenario_spec in config["scenarios"]:
        scenario_name = scenario_spec["name"]
        scenario_cfg = merge_dict(config["defaults"], scenario_spec)
        for seed in config["seeds"]:
            set_seed(seed)
            rng = random.Random(seed)
            train_data = build_dataset(scenario_cfg, "train", rng)
            val_data = build_dataset(scenario_cfg, "val", rng)
            test_data = build_dataset(scenario_cfg, "test", rng)
            solver_cfg = scenario_cfg["solver"]

            for method in config["fixed_methods"]:
                metrics_df, trace_df = evaluate_fixed_baseline(method, solver_cfg, test_data, seed, scenario_name)
                all_metrics.append(metrics_df)
                if not trace_df.empty:
                    all_traces.append(trace_df)

            oracle_metrics, oracle_traces = rollout_oracle(solver_cfg, test_data, seed, scenario_name)
            all_metrics.append(oracle_metrics)
            all_traces.append(oracle_traces)

            for method_cfg in config["learned_methods"]:
                trained = train_policy_variant(method_cfg, solver_cfg, train_data, val_data, seed, device)
                model_path = models_dir / f"{scenario_name}_{method_cfg['name']}_seed{seed}.pt"
                torch.save(trained.model.state_dict(), model_path)
                training_rows.append(
                    {
                        "scenario": scenario_name,
                        "seed": seed,
                        "method": trained.name,
                        "family": trained.family,
                        **trained.training_summary,
                        "model_path": str(model_path.relative_to(run_root)),
                    }
                )
                metrics_df, trace_df = rollout_policy_model(trained, solver_cfg, test_data, seed, scenario_name)
                all_metrics.append(metrics_df)
                all_traces.append(trace_df)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    non_empty_traces = [trace for trace in all_traces if not trace.empty]
    trace_df = pd.concat(non_empty_traces, ignore_index=True) if non_empty_traces else pd.DataFrame()
    training_df = pd.DataFrame(training_rows)
    summary_df = summarize(metrics_df, seed=config["seeds"][0])
    pairwise_df = pairwise_vs_reference(metrics_df, "rl_guided_lbbd")
    verification_df = evaluate_success_criteria(summary_df)

    metrics_df.to_csv(run_root / "metrics.csv", index=False)
    trace_df.to_csv(run_root / "trace.csv", index=False)
    training_df.to_csv(run_root / "training_summary.csv", index=False)
    summary_df.to_csv(run_root / "metrics_summary.csv", index=False)
    pairwise_df.to_csv(run_root / "pairwise_vs_rl.csv", index=False)
    verification_df.to_csv(run_root / "verification.csv", index=False)
    save_plots(summary_df, figures_dir)

    contract = build_contract(config)
    (run_root / "experiment_contract.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    environment = collect_environment()
    environment["torch"] = torch.__version__
    environment["cuda_available"] = torch.cuda.is_available()
    environment["device_used"] = device_name
    environment["git_state"] = git_state(repo_root)
    save_json(run_root / "environment.json", environment)
    (run_root / "git_state.txt").write_text(environment["git_state"], encoding="utf-8")

    commands = [
        f"{sys.executable} -m py_compile experiments/src/*.py",
        f"{sys.executable} scripts/run_first_priority_experiments.py --config {config_path}",
    ]
    (run_root / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    save_json(
        run_root / "data_manifest.json",
        {
            "config_path": str(config_path),
            "metrics_sha256": sha256_file(run_root / "metrics.csv"),
            "trace_sha256": sha256_file(run_root / "trace.csv"),
            "training_sha256": sha256_file(run_root / "training_summary.csv"),
        },
    )

    findings = []
    for scenario in summary_df["scenario"].unique():
        sub = summary_df[summary_df["scenario"] == scenario].set_index("method")
        if {"rl_guided_lbbd", "plain_lbbd", "random_region_lbbd"}.issubset(sub.index):
            findings.append(
                f"{scenario}: rl_guided_lbbd objective_mean={sub.loc['rl_guided_lbbd', 'objective_mean']:.2f}, plain_lbbd={sub.loc['plain_lbbd', 'objective_mean']:.2f}, random_region_lbbd={sub.loc['random_region_lbbd', 'objective_mean']:.2f}."
            )
        if {"rl_guided_lbbd", "rl_no_radius_lbbd", "rl_sparse_reward_lbbd"}.issubset(sub.index):
            findings.append(
                f"{scenario}: full RL vs ablations -> no_radius={sub.loc['rl_no_radius_lbbd', 'objective_mean']:.2f}, sparse_reward={sub.loc['rl_sparse_reward_lbbd', 'objective_mean']:.2f}, full={sub.loc['rl_guided_lbbd', 'objective_mean']:.2f}."
            )
    total_checks = int(len(verification_df))
    passed_checks = int(verification_df["passed"].sum()) if not verification_df.empty else 0
    findings.append(f"success criteria passed {passed_checks}/{total_checks} scenario-level checks.")
    deviations = [
        "This slice completes the first-priority synthetic ID matrix only; public benchmarks and OOD suites remain out of scope.",
        "The learning baseline set is intentionally compact: PPO RL plus oracle behavior cloning.",
        "Confidence intervals are bootstrap CIs over executed runs; paired significance testing is not implemented in this first-priority runner yet.",
    ]
    next_steps = [
        "Add Scale-OOD and Disturbance-OOD as the next execution block from the top-conference plan.",
        "Replace the greedy full_reschedule reference with a true global CP-SAT recomputation baseline.",
        "Expand learning comparators with one heavier neural scheduler only after the decomposition story is stable.",
    ]
    artifact_index = [
        "metrics.csv",
        "metrics_summary.csv",
        "pairwise_vs_rl.csv",
        "verification.csv",
        "trace.csv",
        "training_summary.csv",
        "experiment_contract.yaml",
        "environment.json",
        "git_state.txt",
        "commands.sh",
        "data_manifest.json",
        "figures/",
        "artifacts/",
    ]
    render_report(
        run_root / "report.md",
        contract,
        environment,
        summary_df,
        pairwise_df,
        training_df,
        verification_df,
        commands,
        deviations,
        findings,
        next_steps,
        artifact_index,
    )

    for item in [
        "metrics.csv",
        "metrics_summary.csv",
        "pairwise_vs_rl.csv",
        "verification.csv",
        "trace.csv",
        "training_summary.csv",
        "experiment_contract.yaml",
        "environment.json",
        "git_state.txt",
        "commands.sh",
        "data_manifest.json",
        "report.md",
    ]:
        shutil.copy2(run_root / item, latest_root / item)
    latest_figures = latest_root / "figures"
    latest_artifacts = latest_root / "artifacts"
    if latest_figures.exists():
        shutil.rmtree(latest_figures)
    if latest_artifacts.exists():
        shutil.rmtree(latest_artifacts)
    shutil.copytree(figures_dir, latest_figures)
    shutil.copytree(models_dir, latest_artifacts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
