"""Deterministic scheduler for verifier reports."""

from __future__ import annotations

from typing import Any

from veritas.verifier.enums import ActionType, DetectorMode, SEVERE_VALIDITY_FLAGS
from veritas.verifier.types import SchedulerAction


def choose_next_action(
    verifier_report: dict[str, Any],
    scheduler_state: dict[str, Any],
    pending_cards: list[dict[str, Any]] | None = None,
) -> SchedulerAction:
    pending_cards = pending_cards or []
    budget = scheduler_state.get("budget", {})
    recommended = verifier_report.get("recommended_action", {})
    recommended_type = recommended.get("type")
    global_flags = set(verifier_report.get("global_flags", []))
    detector_mode = verifier_report.get("detector_mode")

    if SEVERE_VALIDITY_FLAGS.intersection(global_flags):
        return SchedulerAction(
            type=ActionType.HUMAN_REVIEW,
            reason="Severe validity flag requires human review.",
            candidate_id=recommended.get("candidate_id"),
        )

    if detector_mode in {
        DetectorMode.INSTRUMENTATION_MISSING.value,
        DetectorMode.SINGLE_SCALAR_NO_SE.value,
    }:
        return SchedulerAction(
            type=ActionType.STOP_UNRESOLVED,
            reason="Detector mode cannot support noise-aware commitment.",
        )

    if (
        recommended_type == ActionType.FINAL_AUDIT.value
        and _audit_budget_remains(budget)
        and _audit_gate_ok(recommended, verifier_report)
    ):
        return SchedulerAction(
            type=ActionType.FINAL_AUDIT,
            reason=recommended.get("reason", "Verifier audit gate passed."),
            candidate_id=recommended.get("candidate_id"),
        )

    if (
        recommended_type == ActionType.RERUN_CANDIDATE.value
        and _rerun_budget_remains(budget)
    ):
        return SchedulerAction(
            type=ActionType.RERUN_CANDIDATE,
            reason=recommended.get("reason", "Verifier requested rerun."),
            candidate_id=recommended.get("candidate_id"),
            seed=recommended.get("seed"),
        )

    if recommended_type == ActionType.REQUEST_NEARBY_VARIANTS.value:
        return SchedulerAction(
            type=ActionType.REQUEST_NEARBY_VARIANTS,
            reason=recommended.get("reason", "Verifier requested nearby variants."),
            candidate_id=recommended.get("candidate_id"),
        )

    runnable_cards = [
        card for card in pending_cards
        if card.get("status", "pending") == "pending"
    ]
    if runnable_cards:
        return SchedulerAction(
            type=ActionType.RUN_PENDING_CARD,
            reason="Runnable pending card is available.",
            card_id=runnable_cards[0].get("card_id"),
        )

    queue_threshold = scheduler_state.get("queue_threshold", 0)
    if len(runnable_cards) < queue_threshold:
        return SchedulerAction(
            type=ActionType.REQUEST_DIVERSE_CARDS,
            reason="Pending queue is below threshold.",
        )

    return SchedulerAction(
        type=ActionType.STOP_UNRESOLVED,
        reason="No runnable scheduler action remains.",
    )


def _audit_budget_remains(budget: dict[str, Any]) -> bool:
    return budget.get("used_audited_candidates", 0) < budget.get("max_audited_candidates", 0)


def _rerun_budget_remains(budget: dict[str, Any]) -> bool:
    return budget.get("used_dev_runs", 0) < budget.get("max_dev_runs", 0)


def _audit_gate_ok(recommended: dict[str, Any], report: dict[str, Any]) -> bool:
    if report.get("detector_mode") != DetectorMode.PER_UNIT_BOOTSTRAP_OK.value:
        return False
    ratio = recommended.get("predicted_audit_resolution_ratio")
    if recommended.get("audit_resolution_ok") is not True and ratio is None:
        return False
    if recommended.get("audit_resolution_ok") is False:
        return False
    z_resolution = recommended.get("z_resolution", 2.0)
    if ratio is not None and ratio < z_resolution:
        return False
    return True
