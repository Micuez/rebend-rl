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
        self.action_space = [(strategy, radius) for strategy in ACTION_STRATEGIES for radius in self.radii]
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
        strategy, radius = self.action_space[action_index]
        unlocked = expand_region(
            self.instance, self.baseline, self.disturbance, strategy, radius, self.rng
        )
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
        if feasible and metrics["objective"] < self.metrics["objective"]:
            self.current = candidate
            self.metrics = metrics
        reward = (old_obj - self.metrics["objective"]) / max(1.0, old_obj) - 0.02
        if not feasible:
            reward -= 0.05
        reward -= 0.01 * elapsed
        self.iteration += 1
        done = self.iteration >= self.max_iterations
        info = {
            "strategy": strategy,
            "radius": radius,
            "feasible": feasible,
            "objective": self.metrics["objective"],
            "note": note,
            "elapsed": elapsed,
        }
        return self.state(), reward, done, info
