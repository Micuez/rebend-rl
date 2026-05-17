from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class OperationSpec:
    op_id: int
    job_id: int
    op_index: int
    eligible_machines: Sequence[int]
    durations: Dict[int, int]


@dataclass(frozen=True)
class Breakdown:
    machine_id: int
    start: int
    end: int


@dataclass(frozen=True)
class Disturbance:
    reschedule_time: int
    breakdown: Breakdown
    inflated_durations: Dict[int, Dict[int, int]]


@dataclass(frozen=True)
class Instance:
    name: str
    num_jobs: int
    num_machines: int
    operations: Sequence[OperationSpec]
    job_op_ids: Dict[int, List[int]]
    due_dates: Dict[int, int]


@dataclass
class ScheduleOperation:
    op_id: int
    machine_id: int
    start: int
    end: int


@dataclass
class Schedule:
    assignments: Dict[int, ScheduleOperation]
    objective: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class RepairResult:
    feasible: bool
    schedule: Schedule
    objective: float
    metrics: Dict[str, float]
    unlocked_count: int
    strategy: str
    radius: int
    solve_time_sec: float
    iteration: int
    note: str = ""
