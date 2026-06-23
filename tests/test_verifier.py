import json

import pytest

from veritas.verifier.cloud import build_cloud_run_manifest, redact_environment
from veritas.verifier.enums import (
    ActionType,
    CardStatus,
    CommandClass,
    DetectorMode,
    UnitOutputStatus,
)
from veritas.verifier.io import append_jsonl, read_jsonl, write_json, write_jsonl
from veritas.verifier.mlrc_adapter import MlrcArtifactError, candidates_from_idea_evals
from veritas.verifier.types import RunRecord


def test_canonical_enum_values_match_docs():
    assert [item.value for item in ActionType] == [
        "run_pending_card",
        "rerun_candidate",
        "request_diverse_cards",
        "request_nearby_variants",
        "abandon_branch",
        "final_audit",
        "stop_success",
        "stop_unresolved",
        "human_review",
    ]
    assert [item.value for item in DetectorMode] == [
        "per_unit_bootstrap_ok",
        "repeated_scalar_seed_se_only",
        "single_scalar_no_se",
        "instrumentation_missing",
    ]
    assert [item.value for item in UnitOutputStatus] == [
        "per_unit_bootstrap_ok",
        "instrumentation_missing",
        "single_scalar_no_se",
        "repeated_scalar_seed_se_only",
        "invalid_unit_outputs",
        "score_recompute_mismatch",
    ]
    assert [item.value for item in CommandClass] == [
        "agent_run",
        "candidate_dev_eval",
        "candidate_audit_eval",
    ]
    assert [item.value for item in CardStatus] == [
        "pending",
        "selected",
        "running",
        "completed",
        "rerun_requested",
        "abandoned",
        "superseded",
        "committed",
    ]


def test_jsonl_helpers_round_trip_dataclasses(tmp_path):
    path = tmp_path / "runs.jsonl"
    assert read_jsonl(path) == []

    first = RunRecord(
        run_id="run_001",
        selected_card_id="card_001",
        status="completed",
        phase="dev",
        seed=0,
        command_class=CommandClass.AGENT_RUN,
        command="bash launch.sh task model 0",
        run_artifact_root="artifacts/run_001",
        candidate_count=1,
    )
    second = RunRecord(
        run_id="run_002",
        selected_card_id=None,
        status="failed",
        phase="dev",
        seed=None,
        command_class=CommandClass.CANDIDATE_DEV_EVAL,
        command="python main.py -m bad -p dev",
        run_artifact_root="artifacts/run_002",
        failure_notes="failed",
    )

    write_jsonl(path, [first])
    append_jsonl(path, second)

    rows = read_jsonl(path)
    assert [row["run_id"] for row in rows] == ["run_001", "run_002"]
    assert rows[0]["command_class"] == "agent_run"
    assert rows[1]["failure_notes"] == "failed"


def test_write_json_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    write_json(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}


def test_cloud_manifest_redacts_sensitive_environment():
    env = {
        "GCP_PROJECT_ID": "veritas-500122",
        "CLOUDSDK_COMPUTE_ZONE": "us-central1-b",
        "GCE_INSTANCE_NAME": "vm-1",
        "CONDA_DEFAULT_ENV": "product-recommendation",
        "MY_OPENAI_API_KEY": "secret-key",
        "MY_AZURE_OPENAI_ENDPOINT": "https://example.invalid",
    }

    redacted = redact_environment(env)
    assert redacted["MY_OPENAI_API_KEY"] == "<redacted>"
    assert redacted["MY_AZURE_OPENAI_ENDPOINT"] == "https://example.invalid"

    manifest = build_cloud_run_manifest(
        "run_001",
        "bash launch.sh product-recommendation gemini-1.5-flash-002 0",
        env,
        git_commit="abc123",
        durable_artifact_uri="gs://bucket/run_001",
    )
    assert manifest.gcp_project_id == "veritas-500122"
    assert manifest.zone == "us-central1-b"
    assert manifest.git_commit == "abc123"
    assert manifest.redacted_environment["MY_OPENAI_API_KEY"] == "<redacted>"


def test_parse_mlrc_idea_evals_into_candidate_records(tmp_path):
    artifact = tmp_path / "idea_evals.json"
    artifact.write_text(
        json.dumps(
            {
                "implementations": [
                    {"method_name": "candidate_a", "performance": 0.71, "step": 3},
                    {"method_name": "candidate_b", "performance": "0.76", "step": 5},
                ]
            }
        )
    )

    candidates = candidates_from_idea_evals(
        artifact,
        run_id="run_009",
        card_id="card_004",
        comparison_origin_id="baseline",
        pool_id="task_dev_baseline_pool_001",
        seed=2,
        log_dir="logs/meta-learning/o1-mini/0622210000_12345",
    )

    assert len(candidates) == 2
    assert [candidate.method_name for candidate in candidates] == [
        "candidate_a",
        "candidate_b",
    ]
    assert [candidate.score for candidate in candidates] == [0.71, 0.76]
    assert candidates[0].score_extracted_by == "runner_wrapper"
    assert candidates[0].llm_score_trusted is False
    assert candidates[0].unit_outputs_status == UnitOutputStatus.SINGLE_SCALAR_NO_SE
    assert candidates[0].validity_flags == ["single_scalar_no_se"]
    assert candidates[0].snapshot_source_path.endswith(
        "env_log/traces/step_3_files"
    )


def test_parse_mlrc_idea_evals_rejects_missing_score(tmp_path):
    artifact = tmp_path / "idea_evals.json"
    artifact.write_text(json.dumps({"implementations": [{"method_name": "bad"}]}))

    with pytest.raises(MlrcArtifactError, match="performance"):
        candidates_from_idea_evals(
            artifact,
            run_id="run_bad",
            card_id=None,
            comparison_origin_id="baseline",
            pool_id="pool",
        )
