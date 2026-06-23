"""File-based verifier pipeline for local ledgers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veritas.verifier.bootstrap import paired_bootstrap
from veritas.verifier.detector import build_scalar_refusal_report, raw_delta
from veritas.verifier.enums import ActionType, DetectorMode, UnitOutputStatus
from veritas.verifier.io import read_json, read_jsonl, to_jsonable, write_json
from veritas.verifier.model import cluster_effective_candidates, rank_candidates
from veritas.verifier.scheduler import choose_next_action
from veritas.verifier.types import CandidateEvidence, CandidateRecord, SchedulerAction
from veritas.verifier.unit_outputs import (
    load_unit_outputs,
    recompute_score_check,
)


DEFAULT_SCHEDULER_STATE = {
    "budget": {
        "max_dev_runs": 10,
        "used_dev_runs": 0,
        "max_reruns_per_candidate": 3,
        "max_audited_candidates": 1,
        "used_audited_candidates": 0,
    },
    "queue_threshold": 0,
}


def run_file_verifier(
    ledger_dir: str | Path,
    *,
    incumbent_id: str = "baseline",
    epsilon: float = 0.01,
    bootstrap_samples: int = 1000,
    seed: int = 0,
    write_outputs: bool = True,
) -> tuple[dict[str, Any], SchedulerAction]:
    ledger_path = Path(ledger_dir)
    candidates = load_candidate_records(ledger_path / "candidates.jsonl")
    incumbent = _find_incumbent(candidates, incumbent_id)

    if _can_run_per_unit(candidates):
        report = build_per_unit_report(
            ledger_path=ledger_path,
            candidates=candidates,
            incumbent=incumbent,
            epsilon=epsilon,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    else:
        report = build_scalar_refusal_report(
            candidates=candidates,
            incumbent=incumbent,
            epsilon=epsilon,
        )

    scheduler_state = _load_optional_json(
        ledger_path / "scheduler_state.json",
        DEFAULT_SCHEDULER_STATE,
    )
    pending_cards = read_jsonl(ledger_path / "intervention_cards.jsonl")
    action = choose_next_action(report, scheduler_state, pending_cards)

    if write_outputs:
        write_json(ledger_path / "verifier_report.json", report)
        write_json(ledger_path / "scheduler_next_action.json", action)

    return report, action


def load_candidate_records(path: str | Path) -> list[CandidateRecord]:
    return [_candidate_from_row(row) for row in read_jsonl(path)]


def build_per_unit_report(
    *,
    ledger_path: Path,
    candidates: list[CandidateRecord],
    incumbent: CandidateRecord,
    epsilon: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    non_incumbents = [
        candidate for candidate in candidates
        if candidate.candidate_id != incumbent.candidate_id
    ]
    incumbent_rows = load_unit_outputs(_resolve_artifact_path(ledger_path, incumbent.unit_outputs_path))
    candidate_rows_by_id = {
        candidate.candidate_id: load_unit_outputs(
            _resolve_artifact_path(ledger_path, candidate.unit_outputs_path)
        )
        for candidate in non_incumbents
    }

    _check_recomputed_scores(candidates, incumbent, incumbent_rows, candidate_rows_by_id)

    bootstrap = paired_bootstrap(
        incumbent_rows=incumbent_rows,
        candidate_rows_by_id=candidate_rows_by_id,
        samples=bootstrap_samples,
        seed=seed,
    )
    evidences = [
        CandidateEvidence(
            candidate_id=candidate.candidate_id,
            logical_candidate_id=candidate.logical_candidate_id,
            delta_hat=bootstrap.delta_hat[candidate.candidate_id],
            total_se=bootstrap.paired_se[candidate.candidate_id],
        )
        for candidate in non_incumbents
    ]
    ranking = rank_candidates(
        cluster_effective_candidates(evidences),
        epsilon=epsilon,
    )
    candidate_reports = _candidate_reports(
        candidates=non_incumbents,
        ranking=ranking,
        bootstrap=bootstrap,
        incumbent=incumbent,
    )
    top_candidate = candidate_reports[0] if candidate_reports else None
    recommended_action = _recommended_action(top_candidate, epsilon)

    return {
        "created_at": _utc_now(),
        "incumbent_candidate_id": incumbent.candidate_id,
        "detector_mode": DetectorMode.PER_UNIT_BOOTSTRAP_OK.value,
        "evidence_mode": "joint_paired_bootstrap",
        "epsilon": epsilon,
        "resolvability": {
            "dev_se_available": True,
            "dev_resolution_ok": bool(top_candidate and top_candidate["dev_resolution_ok"]),
            "audit_se_available": False,
            "predicted_audit_resolution_ok": False,
        },
        "recommended_action": recommended_action,
        "candidates": candidate_reports,
        "uncertainty": {
            "method": "joint_paired_bootstrap",
            "bootstrap_samples": bootstrap_samples,
            "candidate_order": bootstrap.candidate_order,
            "delta_covariance": bootstrap.covariance,
        },
        "audit": {
            "max_audited_candidates": 1,
            "used_audited_candidates": 0,
            "results": [],
        },
        "global_flags": sorted(set(ranking["flags"])),
    }


def _candidate_reports(
    *,
    candidates: list[CandidateRecord],
    ranking: dict[str, Any],
    bootstrap,
    incumbent: CandidateRecord,
) -> list[dict[str, Any]]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ranked_reports: list[dict[str, Any]] = []
    for ranked in ranking["candidates"]:
        member_ids = ranked["member_ids"]
        primary_id = member_ids[0]
        candidate = candidate_by_id[primary_id]
        ranked_reports.append(
            {
                "candidate_id": primary_id,
                "logical_candidate_id": candidate.logical_candidate_id,
                "member_ids": member_ids,
                "raw_score": candidate.score,
                "incumbent_raw_score": incumbent.score,
                "raw_delta": raw_delta(candidate, incumbent),
                "paired_delta": bootstrap.delta_hat[primary_id],
                "paired_se": bootstrap.paired_se[primary_id],
                "total_se": ranked["total_se"],
                "posterior_mean_delta": ranked["posterior_mean_delta"],
                "posterior_sd": ranked["posterior_sd"],
                "lower_evidence_gain": ranked["lower_evidence_gain"],
                "upper_evidence_gain": ranked["upper_evidence_gain"],
                "dev_resolution_ratio": ranked["dev_resolution_ratio"],
                "dev_resolution_ok": ranked["dev_resolution_ok"],
                "uncertainty_high": ranked["uncertainty_high"],
                "decision": _candidate_decision(ranked),
                "unit_outputs_status": str(candidate.unit_outputs_status),
                "flags": ranked["flags"],
            }
        )
    ranked_reports.sort(key=lambda item: item["lower_evidence_gain"], reverse=True)
    return ranked_reports


def _candidate_decision(ranked: dict[str, Any]) -> str:
    if ranked["posterior_mean_delta"] > 0 and ranked["uncertainty_high"]:
        return ActionType.RERUN_CANDIDATE.value
    if ranked["lower_evidence_gain"] > 0 and ranked["dev_resolution_ok"]:
        return ActionType.FINAL_AUDIT.value
    return ActionType.STOP_UNRESOLVED.value


def _recommended_action(top_candidate: dict[str, Any] | None, epsilon: float) -> dict[str, Any]:
    if top_candidate is None:
        return {
            "type": ActionType.STOP_UNRESOLVED.value,
            "reason": "No candidate evidence is available.",
        }
    if top_candidate["lower_evidence_gain"] > 0 and top_candidate["dev_resolution_ok"]:
        ratio = top_candidate["dev_resolution_ratio"]
        return {
            "type": ActionType.FINAL_AUDIT.value,
            "candidate_id": top_candidate["candidate_id"],
            "reason": "Top candidate clears the per-unit evidence gate.",
            "predicted_audit_resolution_ratio": ratio,
            "z_resolution": 2.0,
            "audit_resolution_ok": ratio >= 2.0,
        }
    if top_candidate["posterior_mean_delta"] > 0 and top_candidate["uncertainty_high"]:
        return {
            "type": ActionType.RERUN_CANDIDATE.value,
            "candidate_id": top_candidate["candidate_id"],
            "reason": f"Positive effect but uncertainty remains high for epsilon={epsilon}.",
        }
    return {
        "type": ActionType.STOP_UNRESOLVED.value,
        "candidate_id": top_candidate["candidate_id"],
        "reason": "No candidate clears the evidence gate.",
    }


def _check_recomputed_scores(
    candidates: list[CandidateRecord],
    incumbent: CandidateRecord,
    incumbent_rows: list[dict[str, Any]],
    candidate_rows_by_id: dict[str, list[dict[str, Any]]],
) -> None:
    rows_by_candidate = {
        incumbent.candidate_id: incumbent_rows,
        **candidate_rows_by_id,
    }
    for candidate in candidates:
        if candidate.score is None:
            continue
        rows = rows_by_candidate.get(candidate.candidate_id)
        if rows is None:
            continue
        _, _, status = recompute_score_check(
            rows,
            candidate.score,
            candidate.score_recompute_tolerance or 1e-9,
        )
        if status != UnitOutputStatus.PER_UNIT_BOOTSTRAP_OK:
            raise ValueError(f"score recompute mismatch for {candidate.candidate_id}")


def _can_run_per_unit(candidates: list[CandidateRecord]) -> bool:
    return bool(candidates) and all(
        str(candidate.unit_outputs_status) == UnitOutputStatus.PER_UNIT_BOOTSTRAP_OK.value
        and candidate.unit_outputs_path
        for candidate in candidates
    )


def _find_incumbent(candidates: list[CandidateRecord], incumbent_id: str) -> CandidateRecord:
    for candidate in candidates:
        if candidate.candidate_id == incumbent_id or candidate.logical_candidate_id == incumbent_id:
            return candidate
    raise ValueError(f"incumbent candidate not found: {incumbent_id}")


def _candidate_from_row(row: dict[str, Any]) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=row["candidate_id"],
        logical_candidate_id=row.get("logical_candidate_id", row["candidate_id"]),
        run_id=row.get("run_id", "unknown_run"),
        card_id=row.get("card_id"),
        parent_candidate_id=row.get("parent_candidate_id"),
        comparison_origin_id=row.get("comparison_origin_id", "baseline"),
        pool_id=row.get("pool_id", "default_pool"),
        step=row.get("step"),
        method_name=row.get("method_name", row["candidate_id"]),
        phase=row.get("phase", "dev"),
        seed=row.get("seed"),
        score=row.get("score"),
        score_direction=row.get("score_direction", "maximize"),
        score_source=row.get("score_source", "unknown"),
        score_extracted_by=row.get("score_extracted_by", "runner_wrapper"),
        llm_reported_score=row.get("llm_reported_score"),
        llm_score_trusted=bool(row.get("llm_score_trusted", False)),
        snapshot_path=row.get("snapshot_path", ""),
        unit_outputs_status=row.get("unit_outputs_status", UnitOutputStatus.SINGLE_SCALAR_NO_SE.value),
        snapshot_source_path=row.get("snapshot_source_path"),
        unit_outputs_path=row.get("unit_outputs_path"),
        verifier_only_unit_scores_path=row.get("verifier_only_unit_scores_path"),
        metric_recomputed_score=row.get("metric_recomputed_score"),
        score_recompute_abs_error=row.get("score_recompute_abs_error"),
        score_recompute_tolerance=row.get("score_recompute_tolerance"),
        code_diff_path=row.get("code_diff_path"),
        config_path=row.get("config_path"),
        validity_flags=list(row.get("validity_flags", [])),
    )


def _load_optional_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    return read_json(path)


def _resolve_artifact_path(ledger_path: Path, artifact_path: str | None) -> Path:
    if not artifact_path:
        raise ValueError("artifact path is required")
    path = Path(artifact_path)
    return path if path.is_absolute() else ledger_path / path


def action_to_dict(action: SchedulerAction) -> dict[str, Any]:
    return to_jsonable(action)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
