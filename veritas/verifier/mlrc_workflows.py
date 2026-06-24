"""Reusable MLRC-to-VeriTAS workflow helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veritas.verifier.io import read_json, to_jsonable, write_json, write_jsonl
from veritas.verifier.mlrc_tasks import (
    PRODUCT_RECOMMENDATION_METRICS,
    prepare_mlrc_task_ledger,
)
from veritas.verifier.pipeline import run_file_verifier


@dataclass(frozen=True)
class MlrcImplementation:
    """A single MLRC implementation entry from idea_evals.json."""

    step: int
    method_name: str
    performance: float
    phase: str


@dataclass(frozen=True)
class ProductRunAuditSummary:
    """Summary emitted by the completed-run audit wrapper."""

    run_dir: str
    trial_dir: str
    best_dev: MlrcImplementation
    baseline_prediction_path: str
    candidate_prediction_path: str
    ledger_dir: str
    recommended_action: dict[str, Any]
    scheduler_next_action: dict[str, Any]


def select_best_dev_implementation(idea_evals_path: str | Path) -> MlrcImplementation:
    """Return the best dev implementation in an MLRC idea_evals.json file."""

    payload = read_json(idea_evals_path)
    implementations = payload.get("implementations")
    if not isinstance(implementations, list):
        raise ValueError("idea_evals.json must contain an implementations list")

    dev_rows = [
        row for row in implementations
        if isinstance(row, dict) and row.get("phase") == "dev" and row.get("performance") is not None
    ]
    if not dev_rows:
        raise ValueError("idea_evals.json has no scored dev implementations")

    best = max(dev_rows, key=lambda row: float(row["performance"]))
    if best.get("method_name") in (None, ""):
        raise ValueError("best dev implementation has no method_name")
    if best.get("step") in (None, ""):
        raise ValueError("best dev implementation has no step")

    return MlrcImplementation(
        step=int(best["step"]),
        method_name=str(best["method_name"]),
        performance=float(best["performance"]),
        phase="dev",
    )


def write_product_candidate_specs(
    specs_path: str | Path,
    *,
    baseline_prediction_path: str | Path,
    candidate_prediction_path: str | Path,
    candidate_method_name: str,
    candidate_step: int | None = None,
    candidate_id: str = "best_dev",
    baseline_id: str = "baseline",
    run_id: str | None = None,
) -> Path:
    """Write the two-row candidate spec JSONL used by the product adapter."""

    output_path = Path(specs_path)
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": baseline_id,
            "method_name": "my_method",
            "prediction_path": str(Path(baseline_prediction_path).expanduser()),
            "is_incumbent": True,
            "run_id": run_id,
        },
        {
            "candidate_id": candidate_id,
            "method_name": candidate_method_name,
            "prediction_path": str(Path(candidate_prediction_path).expanduser()),
            "parent_candidate_id": baseline_id,
            "step": candidate_step,
            "run_id": run_id,
        },
    ]
    write_jsonl(output_path, rows)
    return output_path


def run_product_recommendation_verifier(
    *,
    ledger_dir: str | Path,
    candidate_specs_path: str | Path,
    labels_path: str | Path,
    incumbent_id: str = "baseline",
    metric_variant: str = "parsed_mrr",
    epsilon: float = 0.005,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """Prepare product-recommendation ledger artifacts and run the verifier."""

    if metric_variant not in PRODUCT_RECOMMENDATION_METRICS:
        raise ValueError(f"unsupported metric_variant: {metric_variant}")

    prepared = prepare_mlrc_task_ledger(
        task_name="product-recommendation",
        ledger_dir=ledger_dir,
        candidate_specs_path=candidate_specs_path,
        labels_path=labels_path,
        phase="dev",
        metric_variant=metric_variant,
        incumbent_id=incumbent_id,
    )
    report, action = run_file_verifier(
        ledger_dir,
        incumbent_id=incumbent_id,
        epsilon=epsilon,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        write_outputs=True,
    )
    return report, to_jsonable(action), prepared


def build_agent_feedback(verifier_report: dict[str, Any], scheduler_action: dict[str, Any]) -> str:
    """Render a concise text block that can be fed back to an agent/controller."""

    recommended = verifier_report.get("recommended_action", {})
    candidate = _top_candidate(verifier_report)
    lines = [
        "Verifier decision:",
        f"- action: {scheduler_action.get('type') or recommended.get('type')}",
        f"- reason: {scheduler_action.get('reason') or recommended.get('reason')}",
    ]
    candidate_id = scheduler_action.get("candidate_id") or recommended.get("candidate_id")
    if candidate_id:
        lines.append(f"- candidate_id: {candidate_id}")
    if candidate:
        lines.extend(
            [
                f"- raw_delta: {candidate.get('raw_delta')}",
                f"- paired_delta: {candidate.get('paired_delta')}",
                f"- paired_se: {candidate.get('paired_se')}",
                f"- dev_resolution_ratio: {candidate.get('dev_resolution_ratio')}",
                f"- decision: {candidate.get('decision')}",
            ]
        )
    lines.append(
        "Use this as verifier evidence; do not treat it as a new benchmark score."
    )
    return "\n".join(lines) + "\n"


def audit_completed_product_run(
    *,
    mlrc_dir: str | Path,
    run_dir: str | Path,
    trial_dir: str | Path,
    task_python: str | Path,
    epsilon: float = 0.005,
    bootstrap_samples: int = 200,
    seed: int = 0,
    metric_variant: str = "parsed_mrr",
) -> ProductRunAuditSummary:
    """Regenerate predictions for a completed MLRC product run and verify them."""

    mlrc_path = Path(mlrc_dir).expanduser().resolve()
    run_path = Path(run_dir).expanduser().resolve()
    trial_path = Path(trial_dir).expanduser().resolve()
    task_python_path = Path(task_python).expanduser()

    idea_evals_path = run_path / "env_log" / "idea_evals.json"
    best = select_best_dev_implementation(idea_evals_path)

    base_env = mlrc_path / "MLAgentBench" / "benchmarks" / "product-recommendation" / "env"
    snapshot_env = run_path / "env_log" / "traces" / f"step_{best.step}_files"
    if not base_env.exists():
        raise FileNotFoundError(f"MLRC product env not found: {base_env}")
    if not snapshot_env.exists():
        raise FileNotFoundError(f"best-dev snapshot not found: {snapshot_env}")

    preds_dir = trial_path / "preds"
    eval_envs_dir = trial_path / "eval_envs"
    preds_dir.mkdir(parents=True, exist_ok=True)
    eval_envs_dir.mkdir(parents=True, exist_ok=True)

    baseline_env = _materialize_eval_env(
        source_env=base_env,
        target_env=eval_envs_dir / "baseline",
        data_source=base_env / "data",
    )
    candidate_env = _materialize_eval_env(
        source_env=base_env,
        target_env=eval_envs_dir / "best_dev",
        data_source=base_env / "data",
    )
    _overlay_methods(snapshot_env / "methods", candidate_env / "methods")

    baseline_pred = preds_dir / "baseline.csv"
    candidate_pred = preds_dir / f"{best.method_name}.csv"
    _run_method_prediction(
        env_dir=baseline_env,
        method_name="my_method",
        output_prediction_path=baseline_pred,
        task_python=task_python_path,
        mlrc_dir=mlrc_path,
    )
    _run_method_prediction(
        env_dir=candidate_env,
        method_name=best.method_name,
        output_prediction_path=candidate_pred,
        task_python=task_python_path,
        mlrc_dir=mlrc_path,
    )

    specs_path = write_product_candidate_specs(
        trial_path / "candidate_specs.jsonl",
        baseline_prediction_path=baseline_pred,
        candidate_prediction_path=candidate_pred,
        candidate_method_name=best.method_name,
        candidate_step=best.step,
        run_id=run_path.name,
    )
    ledger_dir = trial_path / "ledger"
    report, action, _prepared = run_product_recommendation_verifier(
        ledger_dir=ledger_dir,
        candidate_specs_path=specs_path,
        labels_path=base_env / "data" / "dev_labels.csv",
        incumbent_id="baseline",
        metric_variant=metric_variant,
        epsilon=epsilon,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    feedback = build_agent_feedback(report, action)
    (ledger_dir / "agent_feedback.txt").write_text(feedback, encoding="utf-8")

    summary = ProductRunAuditSummary(
        run_dir=str(run_path),
        trial_dir=str(trial_path),
        best_dev=best,
        baseline_prediction_path=str(baseline_pred),
        candidate_prediction_path=str(candidate_pred),
        ledger_dir=str(ledger_dir),
        recommended_action=report["recommended_action"],
        scheduler_next_action=action,
    )
    write_json(trial_path / "audit_summary.json", summary)
    return summary


def _top_candidate(report: dict[str, Any]) -> dict[str, Any] | None:
    candidates = report.get("candidates") or []
    if not candidates:
        return None
    first = candidates[0]
    return first if isinstance(first, dict) else None


def _materialize_eval_env(*, source_env: Path, target_env: Path, data_source: Path) -> Path:
    if target_env.exists():
        shutil.rmtree(target_env)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {
            name for name in names
            if name in {"data", "output", "backup", "__pycache__"}
        }

    shutil.copytree(source_env, target_env, ignore=ignore, symlinks=True)
    (target_env / "data").symlink_to(data_source, target_is_directory=True)
    return target_env


def _overlay_methods(source_methods: Path, target_methods: Path) -> None:
    if not source_methods.exists():
        raise FileNotFoundError(f"snapshot methods directory not found: {source_methods}")
    target_methods.mkdir(parents=True, exist_ok=True)
    for source in source_methods.iterdir():
        if source.is_file():
            shutil.copy2(source, target_methods / source.name)


def _run_method_prediction(
    *,
    env_dir: Path,
    method_name: str,
    output_prediction_path: Path,
    task_python: Path,
    mlrc_dir: Path,
) -> None:
    output_dir = env_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    env = {**os.environ, "PYTHONPATH": f"{mlrc_dir}:{os.environ.get('PYTHONPATH', '')}"}
    result = subprocess.run(
        [str(task_python), "-u", "main.py", "-m", method_name, "-p", "dev"],
        cwd=env_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"method {method_name!r} failed in {env_dir}")

    pred_path = output_dir / "pred.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"method {method_name!r} did not write {pred_path}")
    output_prediction_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pred_path, output_prediction_path)
