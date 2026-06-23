"""Synthetic file-based demo for the verifier pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veritas.verifier.enums import UnitOutputStatus
from veritas.verifier.io import write_json, write_jsonl
from veritas.verifier.pipeline import run_file_verifier


TRUE_EFFECTS = {
    "cand_stable": 0.05,
    "cand_noisy": 0.01,
}


def create_synthetic_ledger(ledger_dir: str | Path) -> dict[str, Any]:
    ledger_path = Path(ledger_dir)
    unit_dir = ledger_path / "unit_outputs"
    unit_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = _unit_rows("baseline", [0.50] * 20)
    stable_rows = _unit_rows("cand_stable", [0.55] * 20)
    noisy_rows = _unit_rows("cand_noisy", [0.75] * 10 + [0.45] * 10)

    write_jsonl(unit_dir / "baseline.jsonl", baseline_rows)
    write_jsonl(unit_dir / "cand_stable.jsonl", stable_rows)
    write_jsonl(unit_dir / "cand_noisy.jsonl", noisy_rows)

    candidates = [
        _candidate_row("baseline", 0.50, "unit_outputs/baseline.jsonl"),
        _candidate_row("cand_stable", 0.55, "unit_outputs/cand_stable.jsonl"),
        _candidate_row("cand_noisy", 0.60, "unit_outputs/cand_noisy.jsonl"),
    ]
    write_jsonl(ledger_path / "candidates.jsonl", candidates)
    write_json(
        ledger_path / "scheduler_state.json",
        {
            "budget": {
                "max_dev_runs": 10,
                "used_dev_runs": 2,
                "max_reruns_per_candidate": 3,
                "max_audited_candidates": 1,
                "used_audited_candidates": 0,
            },
            "queue_threshold": 0,
        },
    )
    return {
        "true_effects": dict(TRUE_EFFECTS),
        "candidate_scores": {
            "cand_stable": 0.55,
            "cand_noisy": 0.60,
        },
    }


def run_synthetic_demo(
    ledger_dir: str | Path,
    *,
    epsilon: float = 0.02,
    bootstrap_samples: int = 300,
    seed: int = 0,
) -> dict[str, Any]:
    setup = create_synthetic_ledger(ledger_dir)
    report, action = run_file_verifier(
        ledger_dir,
        incumbent_id="baseline",
        epsilon=epsilon,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        write_outputs=True,
    )
    greedy_candidate = max(
        setup["candidate_scores"],
        key=lambda candidate_id: setup["candidate_scores"][candidate_id],
    )
    verifier_candidate = report["candidates"][0]["candidate_id"]
    best_true_candidate = max(
        setup["true_effects"],
        key=lambda candidate_id: setup["true_effects"][candidate_id],
    )

    greedy_regret = setup["true_effects"][best_true_candidate] - setup["true_effects"][greedy_candidate]
    verifier_regret = setup["true_effects"][best_true_candidate] - setup["true_effects"][verifier_candidate]

    summary = {
        "greedy_candidate": greedy_candidate,
        "verifier_candidate": verifier_candidate,
        "best_true_candidate": best_true_candidate,
        "greedy_true_effect": setup["true_effects"][greedy_candidate],
        "verifier_true_effect": setup["true_effects"][verifier_candidate],
        "selective_optimism": setup["candidate_scores"][greedy_candidate]
        - (0.50 + setup["true_effects"][greedy_candidate]),
        "expected_regret": {
            "greedy": greedy_regret,
            "verifier": verifier_regret,
        },
        "near_best_selection": {
            "greedy": greedy_regret <= epsilon,
            "verifier": verifier_regret <= epsilon,
        },
        "audit_drop": {
            "greedy": setup["candidate_scores"][greedy_candidate]
            - (0.50 + setup["true_effects"][greedy_candidate]),
            "verifier": setup["candidate_scores"][verifier_candidate]
            - (0.50 + setup["true_effects"][verifier_candidate]),
        },
        "calibration_checks": {
            "mode": "single_fixture_smoke",
            "note": "Use repeated synthetic simulations for real calibration.",
        },
        "recommended_action": {
            "type": str(action.type),
            "candidate_id": action.candidate_id,
            "reason": action.reason,
        },
    }
    write_json(Path(ledger_dir) / "synthetic_report.json", summary)
    return summary


def _unit_rows(candidate_id: str, values: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "unit_id": f"unit_{index:03d}",
            "unit_type": "synthetic_example",
            "phase": "dev",
            "prediction": candidate_id,
            "target_or_verifier_label": "synthetic",
            "score_component": value,
            "weight": 1.0,
            "metadata": None,
        }
        for index, value in enumerate(values)
    ]


def _candidate_row(candidate_id: str, score: float, unit_outputs_path: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "logical_candidate_id": candidate_id,
        "run_id": "synthetic_run",
        "card_id": None,
        "parent_candidate_id": None if candidate_id == "baseline" else "baseline",
        "comparison_origin_id": "baseline",
        "pool_id": "synthetic_pool",
        "step": None,
        "method_name": candidate_id,
        "phase": "dev",
        "seed": 0,
        "score": score,
        "score_direction": "maximize",
        "score_source": "synthetic",
        "score_extracted_by": "runner_wrapper",
        "llm_reported_score": None,
        "llm_score_trusted": False,
        "snapshot_path": f"snapshots/{candidate_id}",
        "unit_outputs_status": UnitOutputStatus.PER_UNIT_BOOTSTRAP_OK.value,
        "unit_outputs_path": unit_outputs_path,
        "verifier_only_unit_scores_path": unit_outputs_path,
        "metric_recomputed_score": score,
        "score_recompute_abs_error": 0.0,
        "score_recompute_tolerance": 1e-9,
        "code_diff_path": None,
        "config_path": None,
        "validity_flags": [],
    }
