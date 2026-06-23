"""Lightweight verifier data records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from veritas.verifier.enums import ActionType, CommandClass, UnitOutputStatus


@dataclass
class RunRecord:
    run_id: str
    selected_card_id: str | None
    status: str
    phase: str
    seed: int | None
    command_class: CommandClass | str
    command: str
    run_artifact_root: str
    log_dir: str | None = None
    work_dir: str | None = None
    idea_evals_path: str | None = None
    candidate_count: int = 0
    integrity_enforced: bool = False
    protected_digest_before_path: str | None = None
    protected_digest_after_path: str | None = None
    integrity_flags: list[str] = field(default_factory=list)
    runtime_seconds: float | None = None
    failure_notes: str | None = None


@dataclass
class CandidateRecord:
    candidate_id: str
    logical_candidate_id: str
    run_id: str
    card_id: str | None
    parent_candidate_id: str | None
    comparison_origin_id: str
    pool_id: str
    step: int | None
    method_name: str
    phase: str
    seed: int | None
    score: float | None
    score_direction: str
    score_source: str
    score_extracted_by: str
    llm_reported_score: float | None
    llm_score_trusted: bool
    snapshot_path: str
    unit_outputs_status: UnitOutputStatus | str
    snapshot_source_path: str | None = None
    unit_outputs_path: str | None = None
    verifier_only_unit_scores_path: str | None = None
    metric_recomputed_score: float | None = None
    score_recompute_abs_error: float | None = None
    score_recompute_tolerance: float | None = None
    code_diff_path: str | None = None
    config_path: str | None = None
    validity_flags: list[str] = field(default_factory=list)


@dataclass
class SchedulerAction:
    type: ActionType | str
    reason: str
    candidate_id: str | None = None
    card_id: str | None = None
    seed: int | None = None


@dataclass
class CloudRunManifest:
    run_id: str
    command: str
    gcp_project_id: str | None = None
    zone: str | None = None
    instance_name: str | None = None
    machine_type: str | None = None
    accelerator_type: str | None = None
    accelerator_count: int | None = None
    disk_image: str | None = None
    conda_env: str | None = None
    python_version: str | None = None
    cuda_version: str | None = None
    driver_version: str | None = None
    git_commit: str | None = None
    mlrc_commit: str | None = None
    durable_artifact_uri: str | None = None
    redacted_environment: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileDigest:
    path: str
    category: str
    sha256: str


@dataclass
class ProtectedDigestManifest:
    run_id: str
    created_at: str
    digest_algorithm: str
    protected_sources: list[str]
    files: list[FileDigest] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
