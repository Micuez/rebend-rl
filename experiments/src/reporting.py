from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment() -> Dict[str, object]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def git_state(repo_root: Path) -> str:
    try:
        branch = subprocess.check_output(
            ["git", "-C", str(repo_root), "branch", "--show-current"],
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
        )
        return f"branch: {branch}\ncommit: {commit}\nstatus:\n{status}"
    except Exception as exc:
        return f"git_state_unavailable: {exc}"


def save_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    summary = metrics_df.groupby("method")["objective"].mean().sort_values()
    summary.plot(kind="bar")
    plt.ylabel("Mean objective")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def render_report(
    output_path: Path,
    status: str,
    contract: Dict,
    implementation_summary: List[str],
    plan_coverage: List[str],
    environment: Dict,
    data_manifest: Dict,
    commands: List[str],
    results_md: str,
    verification: List[str],
    deviations: List[str],
    limitations: List[str],
    next_steps: List[str],
    artifact_index: List[str],
) -> None:
    lines = [
        f"# Experiment Report",
        "",
        f"## Status",
        status,
        "",
        "## Objective",
        contract["objective"],
        "",
        "## Experiment Contract",
        "```yaml",
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True).strip(),
        "```",
        "",
        "## Implementation Summary",
        *[f"- {item}" for item in implementation_summary],
        "",
        "## README Plan Coverage",
        *[f"- {item}" for item in plan_coverage],
        "",
        "## Environment",
        "```json",
        json.dumps(environment, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Data Manifest",
        "```json",
        json.dumps(data_manifest, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Commands Executed",
        *[f"- `{cmd}`" for cmd in commands],
        "",
        "## Results",
        results_md,
        "",
        "## Verification",
        *[f"- {item}" for item in verification],
        "",
        "## Deviations",
        *[f"- {item}" for item in deviations],
        "",
        "## Limitations",
        *[f"- {item}" for item in limitations],
        "",
        "## Next Steps",
        *[f"- {item}" for item in next_steps],
        "",
        "## Artifact Index",
        *[f"- {item}" for item in artifact_index],
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
