"""Canonical verifier enum values."""

from __future__ import annotations

from enum import Enum


class StringEnum(str, Enum):
    """String-valued enum with readable str() behavior for JSON fields."""

    def __str__(self) -> str:
        return self.value


class ActionType(StringEnum):
    RUN_PENDING_CARD = "run_pending_card"
    RERUN_CANDIDATE = "rerun_candidate"
    REQUEST_DIVERSE_CARDS = "request_diverse_cards"
    REQUEST_NEARBY_VARIANTS = "request_nearby_variants"
    ABANDON_BRANCH = "abandon_branch"
    FINAL_AUDIT = "final_audit"
    STOP_SUCCESS = "stop_success"
    STOP_UNRESOLVED = "stop_unresolved"
    HUMAN_REVIEW = "human_review"


class DetectorMode(StringEnum):
    PER_UNIT_BOOTSTRAP_OK = "per_unit_bootstrap_ok"
    REPEATED_SCALAR_SEED_SE_ONLY = "repeated_scalar_seed_se_only"
    SINGLE_SCALAR_NO_SE = "single_scalar_no_se"
    INSTRUMENTATION_MISSING = "instrumentation_missing"


class UnitOutputStatus(StringEnum):
    PER_UNIT_BOOTSTRAP_OK = "per_unit_bootstrap_ok"
    INSTRUMENTATION_MISSING = "instrumentation_missing"
    SINGLE_SCALAR_NO_SE = "single_scalar_no_se"
    REPEATED_SCALAR_SEED_SE_ONLY = "repeated_scalar_seed_se_only"
    INVALID_UNIT_OUTPUTS = "invalid_unit_outputs"
    SCORE_RECOMPUTE_MISMATCH = "score_recompute_mismatch"


class CommandClass(StringEnum):
    AGENT_RUN = "agent_run"
    CANDIDATE_DEV_EVAL = "candidate_dev_eval"
    CANDIDATE_AUDIT_EVAL = "candidate_audit_eval"


class CardStatus(StringEnum):
    PENDING = "pending"
    SELECTED = "selected"
    RUNNING = "running"
    COMPLETED = "completed"
    RERUN_REQUESTED = "rerun_requested"
    ABANDONED = "abandoned"
    SUPERSEDED = "superseded"
    COMMITTED = "committed"


SEVERE_VALIDITY_FLAGS = frozenset(
    {
        "protected_file_modified",
        "metric_code_modified",
        "data_split_modified",
        "score_recompute_mismatch",
        "invalid_unit_outputs",
        "score_extracted_from_llm_only",
        "integrity_not_enforced",
    }
)

STATISTICAL_BLOCKER_FLAGS = frozenset(
    {
        "instrumentation_missing",
        "single_scalar_no_se",
        "audit_se_unavailable",
        "epsilon_below_noise_floor",
        "audit_below_resolution",
    }
)

CAUTION_FLAGS = frozenset(
    {
        "repeated_scalar_seed_se_only",
        "low_information_uncertainty",
        "low_information_seed_variance",
        "near_duplicate_cluster",
        "low_information_cluster_estimate",
        "cluster_member_selected_posthoc",
        "adaptive_pool_bias",
        "tau2_collapse",
        "predicted_audit_se_shift_risk",
        "hyperparameter_estimator_approximation",
        "independent_posterior_approximation",
    }
)

