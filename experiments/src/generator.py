from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .core import Breakdown, Disturbance, Instance, OperationSpec


def _randint(rng: random.Random, bounds: Tuple[int, int]) -> int:
    return rng.randint(bounds[0], bounds[1])


def generate_instance(
    rng: random.Random,
    name: str,
    job_range: Tuple[int, int],
    num_machines: int,
    ops_per_job: Tuple[int, int],
    flexibility_choices: List[int],
    duration_range: Tuple[int, int],
    due_date_factor: float,
) -> Instance:
    num_jobs = _randint(rng, job_range)
    operations: List[OperationSpec] = []
    job_op_ids: Dict[int, List[int]] = {}
    total_min_duration = 0
    op_id = 0
    for job_id in range(num_jobs):
        n_ops = _randint(rng, ops_per_job)
        job_op_ids[job_id] = []
        for op_index in range(n_ops):
            flex = min(num_machines, rng.choice(flexibility_choices))
            eligible = sorted(rng.sample(list(range(num_machines)), k=flex))
            durations = {
                machine_id: _randint(rng, duration_range) for machine_id in eligible
            }
            total_min_duration += min(durations.values())
            operations.append(
                OperationSpec(
                    op_id=op_id,
                    job_id=job_id,
                    op_index=op_index,
                    eligible_machines=eligible,
                    durations=durations,
                )
            )
            job_op_ids[job_id].append(op_id)
            op_id += 1

    due_dates = {
        job_id: max(1, int(due_date_factor * sum(
            min(operations[op].durations.values()) for op in op_ids
        )))
        for job_id, op_ids in job_op_ids.items()
    }
    horizon_scale = max(total_min_duration // max(1, num_machines), 1)
    due_dates = {job_id: due + horizon_scale for job_id, due in due_dates.items()}
    return Instance(
        name=name,
        num_jobs=num_jobs,
        num_machines=num_machines,
        operations=operations,
        job_op_ids=job_op_ids,
        due_dates=due_dates,
    )


def generate_disturbance(
    rng: random.Random,
    instance: Instance,
    baseline_horizon: int,
    reschedule_fraction: float,
    breakdown_duration_fraction: Tuple[float, float],
    inflation_probability: float,
    inflation_range: Tuple[float, float],
) -> Disturbance:
    reschedule_time = max(1, int(baseline_horizon * reschedule_fraction))
    machine_id = rng.randrange(instance.num_machines)
    frac = rng.uniform(*breakdown_duration_fraction)
    duration = max(1, int(frac * baseline_horizon))
    start = rng.randint(max(0, reschedule_time - duration // 2), max(reschedule_time, baseline_horizon - 1))
    end = min(baseline_horizon + duration, start + duration)
    inflated_durations: Dict[int, Dict[int, int]] = {}
    for op in instance.operations:
        if rng.random() < inflation_probability:
            multiplier = rng.uniform(*inflation_range)
            inflated_durations[op.op_id] = {
                machine_id: max(1, int(round(duration * multiplier)))
                for machine_id, duration in op.durations.items()
            }
    return Disturbance(
        reschedule_time=reschedule_time,
        breakdown=Breakdown(machine_id=machine_id, start=start, end=end),
        inflated_durations=inflated_durations,
    )
