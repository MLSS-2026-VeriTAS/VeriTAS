"""Verifier report builders for early detector modes."""

from __future__ import annotations

from datetime import datetime, timezone

from veritas.verifier.enums import (
    ActionType,
    DetectorMode,
    SEVERE_VALIDITY_FLAGS,
    UnitOutputStatus,
)
from veritas.verifier.types import CandidateRecord


def raw_delta(candidate: CandidateRecord, incumbent: CandidateRecord) -> float | None:
    if candidate.score is None or incumbent.score is None:
        return None
    if candidate.score_direction == "minimize":
        return incumbent.score - candidate.score
    return candidate.score - incumbent.score


def detector_mode_from_candidates(candidates: list[CandidateRecord]) -> DetectorMode:
    statuses = {str(candidate.unit_outputs_status) for candidate in candidates}
    if statuses and statuses <= {UnitOutputStatus.PER_UNIT_BOOTSTRAP_OK.value}:
        return DetectorMode.PER_UNIT_BOOTSTRAP_OK
    if UnitOutputStatus.INSTRUMENTATION_MISSING.value in statuses:
        return DetectorMode.INSTRUMENTATION_MISSING
    if UnitOutputStatus.REPEATED_SCALAR_SEED_SE_ONLY.value in statuses:
        return DetectorMode.REPEATED_SCALAR_SEED_SE_ONLY
    return DetectorMode.SINGLE_SCALAR_NO_SE


def build_scalar_refusal_report(
    *,
    candidates: list[CandidateRecord],
    incumbent: CandidateRecord,
    epsilon: float,
    created_at: str | None = None,
    max_audited_candidates: int = 1,
) -> dict:
    detector_mode = detector_mode_from_candidates(candidates)
    global_flags = _global_flags(candidates, detector_mode)
    action_type = (
        ActionType.HUMAN_REVIEW
        if SEVERE_VALIDITY_FLAGS.intersection(global_flags)
        else ActionType.STOP_UNRESOLVED
    )

    reason = (
        "Only scalar score evidence is available; no standard error is available "
        "for noise adjudication."
    )
    if detector_mode == DetectorMode.INSTRUMENTATION_MISSING:
        reason = "Task adapter or per-unit instrumentation is missing."
    if action_type == ActionType.HUMAN_REVIEW:
        reason = f"{reason} Severe validity flags require human review."

    return {
        "created_at": created_at or _utc_now(),
        "incumbent_candidate_id": incumbent.candidate_id,
        "detector_mode": detector_mode.value,
        "evidence_mode": "scalar_only_refusal",
        "epsilon": epsilon,
        "resolvability": {
            "dev_se_available": False,
            "audit_se_available": False,
            "dev_resolution_ok": False,
            "predicted_audit_resolution_ok": False,
        },
        "recommended_action": {
            "type": action_type.value,
            "reason": reason,
        },
        "candidates": [
            _scalar_candidate_report(candidate, incumbent, action_type)
            for candidate in candidates
            if candidate.candidate_id != incumbent.candidate_id
        ],
        "uncertainty": {
            "method": "none",
        },
        "audit": {
            "max_audited_candidates": max_audited_candidates,
            "used_audited_candidates": 0,
            "results": [],
        },
        "global_flags": sorted(global_flags),
    }


def _scalar_candidate_report(
    candidate: CandidateRecord,
    incumbent: CandidateRecord,
    action_type: ActionType,
) -> dict:
    flags = set(candidate.validity_flags)
    flags.add(str(candidate.unit_outputs_status))
    flags.add("low_information_uncertainty")
    return {
        "candidate_id": candidate.candidate_id,
        "raw_delta": raw_delta(candidate, incumbent),
        "decision": action_type.value,
        "unit_outputs_status": str(candidate.unit_outputs_status),
        "flags": sorted(flags),
    }


def _global_flags(
    candidates: list[CandidateRecord],
    detector_mode: DetectorMode,
) -> set[str]:
    flags = {detector_mode.value, "low_information_uncertainty"}
    for candidate in candidates:
        flags.update(candidate.validity_flags)
        flags.add(str(candidate.unit_outputs_status))
    return flags


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
