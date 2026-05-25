from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .core import Disturbance, Instance, Schedule
from .scheduler import compute_metrics, criticality_scores, expand_region, greedy_schedule, solve_repair


ACTION_STRATEGIES = ["affected_machine", "critical_path", "tardy_job", "random"]


@dataclass
class Transition:
    state: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float


class RepairEnvironment:
    def __init__(
        self,
        instance: Instance,
        baseline: Schedule,
        disturbance: Disturbance,
        radii: Sequence[int],
        weights: Dict[str, float],
        time_limit_sec: float,
        threads: int,
        max_iterations: int,
        seed: int,
        action_strategies: Sequence[str] | None = None,
        reward_config: Dict[str, float] | None = None,
    ) -> None:
        self.instance = instance
        self.baseline = baseline
        self.disturbance = disturbance
        self.radii = list(radii)
        self.weights = weights
        self.time_limit_sec = time_limit_sec
        self.threads = threads
        self.max_iterations = max_iterations
        self.rng = random.Random(seed)
        self.action_strategies = list(action_strategies or ACTION_STRATEGIES)
        self.reward_config = {
            "step_penalty": 0.02,
            "infeasible_penalty": 0.05,
            "elapsed_penalty": 0.01,
        }
        if reward_config:
            self.reward_config.update(reward_config)
        self.action_space = [
            (strategy, radius) for strategy in self.action_strategies for radius in self.radii
        ]
        self.reset()

    def reset(self) -> np.ndarray:
        self.current = greedy_schedule(self.instance, self.disturbance)
        self.metrics = compute_metrics(
            self.instance, self.baseline, self.current, self.disturbance, self.weights
        )
        self.current.objective = self.metrics["objective"]
        self.iteration = 0
        return self.state()

    def state(self) -> np.ndarray:
        scores = criticality_scores(self.instance, self.baseline, self.disturbance)
        critical_values = sorted(scores.values(), reverse=True)
        top_scores = critical_values[:3] + [0.0] * max(0, 3 - len(critical_values))
        base_obj = max(self.metrics["objective"], 1.0)
        features = np.array(
            [
                self.iteration / max(1, self.max_iterations),
                self.metrics["makespan"] / base_obj,
                self.metrics["tardiness"] / base_obj,
                self.metrics["start_deviation"] / base_obj,
                self.metrics["machine_reassignment"] / max(1, len(self.instance.operations)),
                self.disturbance.breakdown.machine_id / max(1, self.instance.num_machines - 1),
                self.disturbance.breakdown.end - self.disturbance.breakdown.start,
                self.disturbance.reschedule_time,
                *top_scores,
            ],
            dtype=np.float32,
        )
        return features

    def step(self, action_index: int):
        preview = self.preview_action(action_index, preserve_rng=False)
        if preview["accepted"]:
            self.current = preview["candidate"]
            self.metrics = preview["metrics"]
        self.iteration += 1
        done = self.iteration >= self.max_iterations
        info = {
            "strategy": preview["strategy"],
            "radius": preview["radius"],
            "feasible": preview["feasible"],
            "objective": self.metrics["objective"],
            "note": preview["note"],
            "elapsed": preview["elapsed"],
            "accepted": preview["accepted"],
        }
        return self.state(), preview["reward"], done, info

    def preview_action(self, action_index: int, preserve_rng: bool = True) -> Dict[str, object]:
        strategy, radius = self.action_space[action_index]
        rng_state = self.rng.getstate()
        unlocked = expand_region(
            self.instance, self.baseline, self.disturbance, strategy, radius, self.rng
        )
        if preserve_rng:
            self.rng.setstate(rng_state)
        feasible, candidate, metrics, elapsed, note = solve_repair(
            self.instance,
            self.baseline,
            self.current,
            self.disturbance,
            unlocked,
            self.weights,
            self.time_limit_sec,
            self.threads,
        )
        old_obj = self.metrics["objective"]
        accepted = bool(feasible and metrics["objective"] < self.metrics["objective"])
        next_metrics = metrics if accepted else self.metrics
        reward = (old_obj - next_metrics["objective"]) / max(1.0, old_obj)
        reward -= self.reward_config["step_penalty"]
        if not feasible:
            reward -= self.reward_config["infeasible_penalty"]
        reward -= self.reward_config["elapsed_penalty"] * elapsed
        return {
            "strategy": strategy,
            "radius": radius,
            "feasible": feasible,
            "objective": next_metrics["objective"],
            "note": note,
            "elapsed": elapsed,
            "accepted": accepted,
            "candidate": candidate,
            "metrics": next_metrics,
            "reward": reward,
        }
