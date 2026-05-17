from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ortools.sat.python import cp_model

from .core import Disturbance, Instance, OperationSpec, RepairResult, Schedule, ScheduleOperation


def op_duration(op: OperationSpec, machine_id: int, disturbance: Optional[Disturbance] = None) -> int:
    if disturbance and op.op_id in disturbance.inflated_durations:
        return disturbance.inflated_durations[op.op_id][machine_id]
    return op.durations[machine_id]


def greedy_schedule(instance: Instance, disturbance: Optional[Disturbance] = None) -> Schedule:
    machine_ready = [0] * instance.num_machines
    job_ready = {job_id: 0 for job_id in instance.job_op_ids}
    assignments: Dict[int, ScheduleOperation] = {}
    for op in instance.operations:
        best = None
        for machine_id in op.eligible_machines:
            start = max(machine_ready[machine_id], job_ready[op.job_id])
            if disturbance and machine_id == disturbance.breakdown.machine_id:
                if start < disturbance.breakdown.end and start + op_duration(op, machine_id, disturbance) > disturbance.breakdown.start:
                    start = disturbance.breakdown.end
            duration = op_duration(op, machine_id, disturbance)
            end = start + duration
            candidate = (end, start, machine_id)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        end, start, machine_id = best
        machine_ready[machine_id] = end
        job_ready[op.job_id] = end
        assignments[op.op_id] = ScheduleOperation(
            op_id=op.op_id, machine_id=machine_id, start=start, end=end
        )
    return Schedule(assignments=assignments)


def compute_metrics(
    instance: Instance,
    baseline: Schedule,
    candidate: Schedule,
    disturbance: Disturbance,
    weights: Dict[str, float],
) -> Dict[str, float]:
    makespan = max(item.end for item in candidate.assignments.values())
    tardiness = 0.0
    for job_id, op_ids in instance.job_op_ids.items():
        completion = max(candidate.assignments[op_id].end for op_id in op_ids)
        tardiness += max(0, completion - instance.due_dates[job_id])
    start_deviation = 0.0
    machine_reassignment = 0.0
    changed_after_td = 0
    total_after_td = 0
    for op in instance.operations:
        base = baseline.assignments[op.op_id]
        new = candidate.assignments[op.op_id]
        if base.end > disturbance.reschedule_time:
            total_after_td += 1
            if base.machine_id != new.machine_id:
                machine_reassignment += 1
            start_deviation += abs(new.start - base.start)
            if base.start != new.start or base.machine_id != new.machine_id:
                changed_after_td += 1
    objective = (
        weights["makespan"] * makespan
        + weights["tardiness"] * tardiness
        + weights["start_deviation"] * start_deviation
        + weights["machine_reassignment"] * machine_reassignment
    )
    return {
        "objective": float(objective),
        "makespan": float(makespan),
        "tardiness": float(tardiness),
        "start_deviation": float(start_deviation),
        "machine_reassignment": float(machine_reassignment),
        "feasible": 1.0,
        "changed_fraction": 0.0 if total_after_td == 0 else changed_after_td / total_after_td,
    }


def build_unfinished_ops(instance: Instance, baseline: Schedule, disturbance: Disturbance) -> Set[int]:
    return {
        op.op_id
        for op in instance.operations
        if baseline.assignments[op.op_id].end > disturbance.reschedule_time
    }


def criticality_scores(
    instance: Instance, baseline: Schedule, disturbance: Disturbance
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for op in instance.operations:
        base = baseline.assignments[op.op_id]
        score = float(base.end - disturbance.reschedule_time)
        if base.machine_id == disturbance.breakdown.machine_id:
            overlap = min(base.end, disturbance.breakdown.end) - max(base.start, disturbance.breakdown.start)
            if overlap > 0:
                score += 5.0 + overlap
        if op.op_id in disturbance.inflated_durations:
            score += 2.0
        scores[op.op_id] = score
    return scores


def expand_region(
    instance: Instance,
    baseline: Schedule,
    disturbance: Disturbance,
    strategy: str,
    radius: int,
    rng,
) -> Set[int]:
    unfinished = build_unfinished_ops(instance, baseline, disturbance)
    scores = criticality_scores(instance, baseline, disturbance)
    breakdown_machine = disturbance.breakdown.machine_id
    affected = [
        op.op_id
        for op in instance.operations
        if baseline.assignments[op.op_id].machine_id == breakdown_machine
        and baseline.assignments[op.op_id].end > disturbance.breakdown.start
        and baseline.assignments[op.op_id].start < disturbance.breakdown.end
    ]
    ranked: List[int]
    if strategy == "critical_path":
        ranked = sorted(unfinished, key=lambda op_id: scores[op_id], reverse=True)
    elif strategy == "affected_machine":
        ranked = affected + [op_id for op_id in unfinished if op_id not in set(affected)]
    elif strategy == "tardy_job":
        tardy_jobs = sorted(
            instance.job_op_ids,
            key=lambda job_id: max(
                0,
                baseline.assignments[instance.job_op_ids[job_id][-1]].end
                - instance.due_dates[job_id],
            ),
            reverse=True,
        )
        ranked = []
        for job_id in tardy_jobs:
            ranked.extend(op_id for op_id in instance.job_op_ids[job_id] if op_id in unfinished)
    elif strategy == "random":
        ranked = list(unfinished)
        rng.shuffle(ranked)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    unlocked: Set[int] = set()
    frontier = ranked[: max(1, radius)]
    unlocked.update(frontier)
    for op_id in list(frontier):
        op = instance.operations[op_id]
        unlocked.update(
            other
            for other in instance.job_op_ids[op.job_id]
            if other in unfinished and abs(instance.operations[other].op_index - op.op_index) <= 1
        )
    return unlocked & unfinished


def solve_repair(
    instance: Instance,
    baseline: Schedule,
    current: Schedule,
    disturbance: Disturbance,
    unlocked: Set[int],
    weights: Dict[str, float],
    time_limit_sec: float,
    threads: int,
) -> Tuple[bool, Schedule, Dict[str, float], float, str]:
    start_time = time.time()
    model = cp_model.CpModel()
    horizon = max(
        max(item.end for item in baseline.assignments.values()) + 50,
        disturbance.breakdown.end + 50,
    )
    unfinished = build_unfinished_ops(instance, baseline, disturbance)
    variables: Dict[int, Dict[str, object]] = {}
    machine_intervals: Dict[int, List[cp_model.IntervalVar]] = defaultdict(list)

    for op in instance.operations:
        base = current.assignments[op.op_id]
        if op.op_id not in unfinished or op.op_id not in unlocked:
            if base.machine_id == disturbance.breakdown.machine_id:
                if base.start < disturbance.breakdown.end and base.end > disturbance.breakdown.start:
                    return False, current, {}, time.time() - start_time, "frozen operation overlaps breakdown"
            variables[op.op_id] = {
                "start": base.start,
                "end": base.end,
                "machine": base.machine_id,
            }
            continue

        start = model.NewIntVar(disturbance.reschedule_time, horizon, f"s_{op.op_id}")
        end = model.NewIntVar(disturbance.reschedule_time, horizon, f"e_{op.op_id}")
        presences = {}
        starts = {}
        ends = {}
        intervals = {}
        for machine_id in op.eligible_machines:
            duration = op_duration(op, machine_id, disturbance)
            s_var = model.NewIntVar(disturbance.reschedule_time, horizon, f"s_{op.op_id}_{machine_id}")
            e_var = model.NewIntVar(disturbance.reschedule_time, horizon, f"e_{op.op_id}_{machine_id}")
            present = model.NewBoolVar(f"y_{op.op_id}_{machine_id}")
            interval = model.NewOptionalIntervalVar(
                s_var, duration, e_var, present, f"i_{op.op_id}_{machine_id}"
            )
            presences[machine_id] = present
            starts[machine_id] = s_var
            ends[machine_id] = e_var
            intervals[machine_id] = interval
            machine_intervals[machine_id].append(interval)
        model.AddExactlyOne(presences.values())
        for machine_id in op.eligible_machines:
            model.Add(start == starts[machine_id]).OnlyEnforceIf(presences[machine_id])
            model.Add(end == ends[machine_id]).OnlyEnforceIf(presences[machine_id])
        variables[op.op_id] = {
            "start": start,
            "end": end,
            "machine_presence": presences,
            "machine": None,
            "intervals": intervals,
        }

    for machine_id in range(instance.num_machines):
        fixed_blocks = []
        for op_id, info in variables.items():
            if isinstance(info["start"], int) and info["machine"] == machine_id:
                fixed_blocks.append((int(info["start"]), int(info["end"])))
        if machine_id == disturbance.breakdown.machine_id:
            fixed_blocks.append((disturbance.breakdown.start, disturbance.breakdown.end))
        for block_start, block_end in fixed_blocks:
            interval = model.NewIntervalVar(
                block_start,
                max(1, block_end - block_start),
                block_end,
                f"fixed_{machine_id}_{block_start}_{block_end}",
            )
            machine_intervals[machine_id].append(interval)
        model.AddNoOverlap(machine_intervals[machine_id])

    for job_id, op_ids in instance.job_op_ids.items():
        for prev, nxt in zip(op_ids[:-1], op_ids[1:]):
            prev_end = variables[prev]["end"]
            next_start = variables[nxt]["start"]
            model.Add(next_start >= prev_end)

    deviations = []
    reassignments = []
    makespan = model.NewIntVar(0, horizon, "makespan")
    tardiness_vars = []
    for op in instance.operations:
        info = variables[op.op_id]
        end_var = info["end"]
        base = baseline.assignments[op.op_id]
        if not isinstance(info["start"], int):
            delta = model.NewIntVar(0, horizon, f"dev_{op.op_id}")
            model.AddAbsEquality(delta, info["start"] - base.start)
            deviations.append(delta)
            if "machine_presence" in info:
                stay = info["machine_presence"].get(base.machine_id)
                if stay is not None:
                    changed = model.NewIntVar(0, 1, f"chg_{op.op_id}")
                    model.Add(changed == 1 - stay)
                    reassignments.append(changed)
                else:
                    reassignments.append(model.NewConstant(1))
        job_last = instance.job_op_ids[op.job_id][-1]
        if op.op_id == job_last:
            tardiness = model.NewIntVar(0, horizon, f"tard_{op.job_id}")
            model.Add(tardiness >= end_var - instance.due_dates[op.job_id])
            model.Add(tardiness >= 0)
            tardiness_vars.append(tardiness)
            model.Add(makespan >= end_var)
    objective_terms = [
        int(weights["makespan"] * 100) * makespan,
        int(weights["tardiness"] * 100) * sum(tardiness_vars),
        int(weights["start_deviation"] * 100) * sum(deviations),
        int(weights["machine_reassignment"] * 100) * sum(reassignments),
    ]
    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.num_search_workers = threads
    solver.parameters.random_seed = 0
    status = solver.Solve(model)
    elapsed = time.time() - start_time
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False, current, {}, elapsed, solver.StatusName(status)

    assignments: Dict[int, ScheduleOperation] = {}
    for op in instance.operations:
        info = variables[op.op_id]
        if isinstance(info["start"], int):
            machine_id = int(info["machine"])
            start = int(info["start"])
            end = int(info["end"])
        else:
            machine_id = next(
                machine_id
                for machine_id, var in info["machine_presence"].items()
                if solver.Value(var) == 1
            )
            start = solver.Value(info["start"])
            end = solver.Value(info["end"])
        assignments[op.op_id] = ScheduleOperation(
            op_id=op.op_id, machine_id=machine_id, start=start, end=end
        )
    schedule = Schedule(assignments=assignments)
    metrics = compute_metrics(instance, baseline, schedule, disturbance, weights)
    schedule.objective = metrics["objective"]
    schedule.metadata = metrics
    return True, schedule, metrics, elapsed, solver.StatusName(status)


def run_policy_episode(
    instance: Instance,
    baseline: Schedule,
    disturbance: Disturbance,
    weights: Dict[str, float],
    action_sequence: Sequence[Tuple[str, int]],
    time_limit_sec: float,
    threads: int,
) -> Tuple[Schedule, List[RepairResult]]:
    current = greedy_schedule(instance, disturbance)
    current_metrics = compute_metrics(instance, baseline, current, disturbance, weights)
    current.objective = current_metrics["objective"]
    history: List[RepairResult] = []
    for iteration, (strategy, radius) in enumerate(action_sequence):
        unlocked = expand_region(
            instance, baseline, disturbance, strategy=strategy, radius=radius, rng=__import__("random")
        )
        feasible, candidate, metrics, elapsed, note = solve_repair(
            instance=instance,
            baseline=baseline,
            current=current,
            disturbance=disturbance,
            unlocked=unlocked,
            weights=weights,
            time_limit_sec=time_limit_sec,
            threads=threads,
        )
        if feasible and metrics["objective"] < current_metrics["objective"]:
            current = candidate
            current_metrics = metrics
        history.append(
            RepairResult(
                feasible=feasible,
                schedule=current,
                objective=current_metrics["objective"],
                metrics=current_metrics,
                unlocked_count=len(unlocked),
                strategy=strategy,
                radius=radius,
                solve_time_sec=elapsed,
                iteration=iteration,
                note=note,
            )
        )
    current.objective = current_metrics["objective"]
    current.metadata = current_metrics
    return current, history
