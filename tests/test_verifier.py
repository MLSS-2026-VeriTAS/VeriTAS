import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from veritas.verifier.bootstrap import paired_bootstrap
from veritas.verifier.cloud import build_cloud_run_manifest, redact_environment
from veritas.verifier.detector import build_scalar_refusal_report, raw_delta
from veritas.verifier.enums import (
    ActionType,
    CardStatus,
    CommandClass,
    DetectorMode,
    UnitOutputStatus,
)
from veritas.verifier.integrity import (
    build_protected_digest_manifest,
    compare_digest_manifests,
    load_protected_digest_manifest,
    write_protected_digest_manifest,
)
from veritas.verifier.io import append_jsonl, read_jsonl, write_json, write_jsonl
from veritas.verifier.mlrc_adapter import MlrcArtifactError, candidates_from_idea_evals
from veritas.verifier.mlrc_tasks import (
    MlrcTaskAdapterError,
    prepare_mlrc_task_ledger,
    product_recommendation_unit_outputs,
)
from veritas.verifier.mlrc_workflows import (
    build_agent_feedback,
    select_best_dev_implementation,
    write_product_candidate_specs,
)
from veritas.verifier.model import (
    cluster_effective_candidates,
    estimate_eb_parameters,
    rank_candidates,
)
from veritas.verifier.scheduler import choose_next_action
from veritas.verifier.types import CandidateEvidence, CandidateRecord, RunRecord
from veritas.verifier.unit_outputs import (
    UnitOutputError,
    align_unit_outputs,
    load_unit_outputs,
    recompute_mean_score,
    recompute_score_check,
)
from scripts.apply_mlrc_veritas_hook_patch import patch_low_level_actions


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


def test_integrity_manifest_round_trip_and_clean_compare(tmp_path):
    task_root = tmp_path / "task"
    (task_root / "data").mkdir(parents=True)
    (task_root / "evaluation.py").write_text("metric = 1\n")
    (task_root / "data" / "dev.json").write_text("{}\n")
    (task_root / "read_only_files.txt").write_text("data/dev.json\n")

    manifest = build_protected_digest_manifest(
        run_id="run_001",
        task_root=task_root,
        protected_paths=["data"],
        metric_paths=["evaluation.py"],
        split_paths=["data/dev.json"],
        read_only_file=task_root / "read_only_files.txt",
        created_at="2026-06-21T00:00:00Z",
    )

    assert manifest.missing_files == []
    assert {item.category for item in manifest.files} == {"metric_code", "data_split"}
    assert compare_digest_manifests(manifest, manifest) == []

    path = tmp_path / "manifest.json"
    write_protected_digest_manifest(path, manifest)
    loaded = load_protected_digest_manifest(path)
    assert loaded.run_id == "run_001"
    assert loaded.files[0].sha256 == manifest.files[0].sha256


def test_integrity_compare_maps_changes_to_flags(tmp_path):
    task_root = tmp_path / "task"
    (task_root / "data").mkdir(parents=True)
    (task_root / "protected").mkdir()
    (task_root / "evaluation.py").write_text("metric = 1\n")
    (task_root / "data" / "dev.json").write_text("{}\n")
    (task_root / "protected" / "config.txt").write_text("ok\n")

    before = build_protected_digest_manifest(
        run_id="run_001",
        task_root=task_root,
        protected_paths=["protected/config.txt"],
        metric_paths=["evaluation.py"],
        split_paths=["data"],
        created_at="2026-06-21T00:00:00Z",
    )

    (task_root / "evaluation.py").write_text("metric = 2\n")
    (task_root / "data" / "dev.json").unlink()
    (task_root / "protected" / "config.txt").unlink()

    after = build_protected_digest_manifest(
        run_id="run_001",
        task_root=task_root,
        protected_paths=["protected/config.txt"],
        metric_paths=["evaluation.py"],
        split_paths=["data"],
        created_at="2026-06-21T00:01:00Z",
    )

    assert compare_digest_manifests(before, after) == [
        "data_split_modified",
        "metric_code_modified",
        "protected_file_modified",
    ]


def test_integrity_unresolved_protected_set_flags_not_enforced(tmp_path):
    task_root = tmp_path / "task"
    task_root.mkdir()
    manifest = build_protected_digest_manifest(
        run_id="run_001",
        task_root=task_root,
        protected_paths=["missing.txt"],
        created_at="2026-06-21T00:00:00Z",
    )

    assert manifest.missing_files == ["missing.txt"]
    assert compare_digest_manifests(manifest, manifest) == ["integrity_not_enforced"]


def test_scalar_refusal_report_does_not_fabricate_uncertainty():
    incumbent = _candidate("baseline", 0.70)
    candidate = _candidate("cand_001", 0.76)

    report = build_scalar_refusal_report(
        candidates=[incumbent, candidate],
        incumbent=incumbent,
        epsilon=0.01,
        created_at="2026-06-21T00:00:00Z",
    )

    assert report["detector_mode"] == "single_scalar_no_se"
    assert report["recommended_action"]["type"] == "stop_unresolved"
    assert report["resolvability"]["dev_se_available"] is False
    assert report["uncertainty"]["method"] == "none"
    assert report["candidates"][0]["raw_delta"] == pytest.approx(0.06)
    assert "paired_se" not in report["candidates"][0]
    assert "model_prob_best" not in report["candidates"][0]
    assert "audit_resolution_ratio" not in report


def test_scalar_refusal_routes_severe_flags_to_human_review():
    incumbent = _candidate("baseline", 0.70)
    candidate = _candidate(
        "cand_001",
        0.76,
        validity_flags=["protected_file_modified"],
    )

    report = build_scalar_refusal_report(
        candidates=[incumbent, candidate],
        incumbent=incumbent,
        epsilon=0.01,
        created_at="2026-06-21T00:00:00Z",
    )

    assert report["recommended_action"]["type"] == "human_review"
    assert "protected_file_modified" in report["global_flags"]
    assert report["candidates"][0]["decision"] == "human_review"


def test_raw_delta_respects_minimize_direction():
    incumbent = _candidate("baseline", 0.50, score_direction="minimize")
    candidate = _candidate("cand_001", 0.40, score_direction="minimize")
    assert raw_delta(candidate, incumbent) == pytest.approx(0.10)


def test_unit_outputs_load_align_and_recompute(tmp_path):
    path = tmp_path / "unit_outputs.jsonl"
    write_jsonl(
        path,
        [
            {
                "unit_id": "u1",
                "unit_type": "example",
                "phase": "dev",
                "prediction": "a",
                "target_or_verifier_label": "a",
                "score_component": None,
            },
            {
                "unit_id": "u2",
                "unit_type": "example",
                "phase": "dev",
                "prediction": "b",
                "target_or_verifier_label": "a",
                "score_component": None,
            },
        ],
    )

    rows = load_unit_outputs(path)
    assert recompute_mean_score(rows) == pytest.approx(0.5)
    recomputed, abs_error, status = recompute_score_check(rows, 0.5, 1e-9)
    assert recomputed == pytest.approx(0.5)
    assert abs_error == pytest.approx(0.0)
    assert status == UnitOutputStatus.PER_UNIT_BOOTSTRAP_OK

    with pytest.raises(UnitOutputError, match="unit_id mismatch"):
        align_unit_outputs(rows, rows[:1])


def test_recompute_score_mismatch_returns_status():
    rows = [
        {"unit_id": "u1", "score_component": 1.0},
        {"unit_id": "u2", "score_component": 0.0},
    ]
    _, abs_error, status = recompute_score_check(rows, 0.9, 0.01)
    assert abs_error == pytest.approx(0.4)
    assert status == UnitOutputStatus.SCORE_RECOMPUTE_MISMATCH


def test_paired_bootstrap_produces_covariance_for_aligned_units():
    incumbent = [
        {"unit_id": "u1", "score_component": 0.0},
        {"unit_id": "u2", "score_component": 0.0},
        {"unit_id": "u3", "score_component": 1.0},
        {"unit_id": "u4", "score_component": 1.0},
    ]
    candidate_a = [
        {"unit_id": "u1", "score_component": 1.0},
        {"unit_id": "u2", "score_component": 0.0},
        {"unit_id": "u3", "score_component": 1.0},
        {"unit_id": "u4", "score_component": 1.0},
    ]
    candidate_b = [
        {"unit_id": "u1", "score_component": 1.0},
        {"unit_id": "u2", "score_component": 1.0},
        {"unit_id": "u3", "score_component": 0.0},
        {"unit_id": "u4", "score_component": 1.0},
    ]

    result = paired_bootstrap(
        incumbent_rows=incumbent,
        candidate_rows_by_id={"cand_a": candidate_a, "cand_b": candidate_b},
        score_direction="maximize",
        samples=200,
        seed=123,
    )

    assert result.candidate_order == ["cand_a", "cand_b"]
    assert result.delta_hat["cand_a"] == pytest.approx(0.25, abs=0.06)
    assert result.delta_hat["cand_b"] == pytest.approx(0.25, abs=0.08)
    assert result.paired_se["cand_a"] > 0
    assert len(result.covariance) == 2
    assert len(result.covariance[0]) == 2


def test_paired_bootstrap_rejects_unaligned_candidate_units():
    incumbent = [{"unit_id": "u1", "score_component": 0.0}]
    candidate = [{"unit_id": "u2", "score_component": 1.0}]
    with pytest.raises(ValueError, match="unit IDs do not align"):
        paired_bootstrap(
            incumbent_rows=incumbent,
            candidate_rows_by_id={"cand": candidate},
            score_direction="maximize",
            samples=10,
        )


def test_paired_bootstrap_respects_minimize_direction():
    incumbent = [
        {"unit_id": "u1", "score_component": 0.5},
        {"unit_id": "u2", "score_component": 0.5},
    ]
    candidate = [
        {"unit_id": "u1", "score_component": 0.4},
        {"unit_id": "u2", "score_component": 0.4},
    ]

    result = paired_bootstrap(
        incumbent_rows=incumbent,
        candidate_rows_by_id={"candidate": candidate},
        score_direction="minimize",
        samples=20,
    )

    assert result.delta_hat["candidate"] == pytest.approx(0.1)


def test_eb_clustering_and_ranking_guards_tiny_pools():
    evidences = [
        CandidateEvidence("cand_a_seed1", "cand_a", 0.020, 0.010),
        CandidateEvidence("cand_a_seed2", "cand_a", 0.030, 0.010),
        CandidateEvidence("cand_b", "cand_b", 0.015, 0.010),
    ]

    effective = cluster_effective_candidates(evidences)
    assert len(effective) == 2
    cand_a = next(candidate for candidate in effective if candidate.cluster_id == "cand_a")
    assert cand_a.member_ids == ["cand_a_seed1", "cand_a_seed2"]
    assert "near_duplicate_cluster" in cand_a.flags

    params = estimate_eb_parameters(effective)
    assert "eb_ranking_only" in params["flags"]

    ranking = rank_candidates(effective, epsilon=0.005)
    assert "eb_ranking_only" in ranking["flags"]
    assert ranking["candidates"][0]["posterior_mean_delta"] >= ranking["candidates"][1]["posterior_mean_delta"]


def test_eb_tau2_collapse_is_flagged():
    effective = [
        CandidateEvidence("cand_a", "cand_a", 0.010, 0.050),
        CandidateEvidence("cand_b", "cand_b", 0.011, 0.050),
    ]
    clustered = cluster_effective_candidates(effective)
    ranking = rank_candidates(clustered, epsilon=0.01)
    assert "tau2_collapse" in ranking["flags"]
    assert "tau2_collapse" in ranking["candidates"][0]["flags"]


def test_scheduler_scalar_refusal_stops_unresolved():
    action = choose_next_action(
        {
            "detector_mode": "single_scalar_no_se",
            "recommended_action": {"type": "stop_unresolved"},
            "global_flags": ["single_scalar_no_se"],
        },
        _scheduler_state(),
    )
    assert action.type == ActionType.STOP_UNRESOLVED


def test_scheduler_routes_severe_flags_to_human_review():
    action = choose_next_action(
        {
            "detector_mode": "per_unit_bootstrap_ok",
            "recommended_action": {
                "type": "final_audit",
                "candidate_id": "cand_001",
            },
            "global_flags": ["protected_file_modified"],
        },
        _scheduler_state(),
    )
    assert action.type == ActionType.HUMAN_REVIEW
    assert action.candidate_id == "cand_001"


def test_scheduler_final_audit_gate_passes():
    action = choose_next_action(
        {
            "detector_mode": "per_unit_bootstrap_ok",
            "recommended_action": {
                "type": "final_audit",
                "candidate_id": "cand_001",
                "audit_resolution_ok": True,
                "predicted_audit_resolution_ratio": 2.5,
                "z_resolution": 2.0,
            },
            "global_flags": [],
        },
        _scheduler_state(),
    )
    assert action.type == ActionType.FINAL_AUDIT
    assert action.candidate_id == "cand_001"


def test_scheduler_final_audit_gate_blocks_missing_resolution():
    action = choose_next_action(
        {
            "detector_mode": "per_unit_bootstrap_ok",
            "recommended_action": {
                "type": "final_audit",
                "candidate_id": "cand_001",
            },
            "global_flags": [],
        },
        _scheduler_state(queue_threshold=0),
    )
    assert action.type == ActionType.STOP_UNRESOLVED


def test_scheduler_rerun_and_nearby_and_pending_paths():
    rerun = choose_next_action(
        {
            "detector_mode": "per_unit_bootstrap_ok",
            "recommended_action": {
                "type": "rerun_candidate",
                "candidate_id": "cand_001",
                "seed": 3,
            },
            "global_flags": [],
        },
        _scheduler_state(),
    )
    assert rerun.type == ActionType.RERUN_CANDIDATE
    assert rerun.seed == 3

    nearby = choose_next_action(
        {
            "detector_mode": "per_unit_bootstrap_ok",
            "recommended_action": {
                "type": "request_nearby_variants",
                "candidate_id": "cand_001",
            },
            "global_flags": [],
        },
        _scheduler_state(),
    )
    assert nearby.type == ActionType.REQUEST_NEARBY_VARIANTS

    pending = choose_next_action(
        {
            "detector_mode": "per_unit_bootstrap_ok",
            "recommended_action": {"type": "stop_unresolved"},
            "global_flags": [],
        },
        _scheduler_state(),
        pending_cards=[{"card_id": "card_001", "status": "pending"}],
    )
    assert pending.type == ActionType.RUN_PENDING_CARD
    assert pending.card_id == "card_001"


def test_scheduler_requests_diverse_cards_when_queue_is_empty():
    action = choose_next_action(
        {
            "detector_mode": "per_unit_bootstrap_ok",
            "recommended_action": {"type": "stop_unresolved"},
            "global_flags": [],
        },
        _scheduler_state(queue_threshold=2),
        pending_cards=[],
    )
    assert action.type == ActionType.REQUEST_DIVERSE_CARDS


def test_file_based_cli_writes_scalar_refusal_outputs(tmp_path):
    write_jsonl(
        tmp_path / "candidates.jsonl",
        [
            _candidate("baseline", 0.70),
            _candidate("cand_001", 0.76),
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_verifier.py",
            "--ledger-dir",
            str(tmp_path),
            "--epsilon",
            "0.01",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "verifier_report.json").read_text())
    next_action = json.loads((tmp_path / "scheduler_next_action.json").read_text())
    assert report["detector_mode"] == "single_scalar_no_se"
    assert report["recommended_action"]["type"] == "stop_unresolved"
    assert next_action["type"] == "stop_unresolved"


def test_synthetic_demo_cli_exercises_per_unit_detector(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_verifier.py",
            "--ledger-dir",
            str(tmp_path),
            "--synthetic-demo",
            "--epsilon",
            "0.02",
            "--bootstrap-samples",
            "200",
            "--seed",
            "123",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "verifier_report.json").read_text())
    synthetic_report = json.loads((tmp_path / "synthetic_report.json").read_text())

    assert report["detector_mode"] == "per_unit_bootstrap_ok"
    assert synthetic_report["greedy_candidate"] == "cand_noisy"
    assert synthetic_report["verifier_candidate"] == "cand_stable"
    assert synthetic_report["expected_regret"]["verifier"] < synthetic_report["expected_regret"]["greedy"]
    assert "calibration_checks" in synthetic_report


def test_product_recommendation_units_recompute_parsed_mrr():
    predictions = [
        {"session_id": "s1", "locale": "US", "next_item_prediction": "['a', 'b', 'c']"},
        {"session_id": "s2", "locale": "US", "next_item_prediction": "['c', 'd']"},
        {"session_id": "s3", "locale": "DE", "next_item_prediction": "['x', 'y']"},
    ]
    labels = [
        {"session_id": "s1", "next_item": "b"},
        {"session_id": "s2", "next_item": "c"},
        {"session_id": "s3", "next_item": "z"},
    ]

    rows = product_recommendation_unit_outputs(
        predictions=predictions,
        labels=labels,
        candidate_id="cand_001",
        phase="dev",
        metric_variant="parsed_mrr",
    )

    assert [row["unit_id"] for row in rows] == ["s1", "s2", "s3"]
    assert [row["score_component"] for row in rows] == [0.5, 1.0, 0.0]
    assert recompute_mean_score(rows) == pytest.approx(0.5)
    assert rows[0]["metadata"]["rank"] == 2

    exact_rows = product_recommendation_unit_outputs(
        predictions=predictions[:1],
        labels=labels[:1],
        candidate_id="cand_001",
        phase="dev",
        metric_variant="mlrc_exact",
    )
    assert exact_rows[0]["score_component"] == pytest.approx(1 / ("['a', 'b', 'c']".index("b") + 1))


def test_product_recommendation_adapter_prepares_verifier_ledger(tmp_path):
    labels_path = tmp_path / "labels.csv"
    baseline_path = tmp_path / "baseline_pred.csv"
    candidate_path = tmp_path / "candidate_pred.csv"
    specs_path = tmp_path / "candidate_specs.jsonl"
    ledger_dir = tmp_path / "ledger"

    _write_csv(
        labels_path,
        [
            {"session_id": "s1", "next_item": "b"},
            {"session_id": "s2", "next_item": "c"},
            {"session_id": "s3", "next_item": "d"},
            {"session_id": "s4", "next_item": "a"},
        ],
    )
    _write_csv(
        baseline_path,
        [
            {"session_id": "s1", "locale": "US", "next_item_prediction": "['x', 'b']"},
            {"session_id": "s2", "locale": "US", "next_item_prediction": "['z', 'y', 'c']"},
            {"session_id": "s3", "locale": "DE", "next_item_prediction": "['q']"},
            {"session_id": "s4", "locale": "JP", "next_item_prediction": "['a']"},
        ],
    )
    _write_csv(
        candidate_path,
        [
            {"session_id": "s1", "locale": "US", "next_item_prediction": "['b']"},
            {"session_id": "s2", "locale": "US", "next_item_prediction": "['c']"},
            {"session_id": "s3", "locale": "DE", "next_item_prediction": "['d']"},
            {"session_id": "s4", "locale": "JP", "next_item_prediction": "['z', 'a']"},
        ],
    )
    write_jsonl(
        specs_path,
        [
            {
                "candidate_id": "baseline",
                "method_name": "my_method",
                "prediction_path": str(baseline_path),
                "is_incumbent": True,
            },
            {
                "candidate_id": "cand_better",
                "method_name": "better_method",
                "prediction_path": str(candidate_path),
            },
        ],
    )

    prepared = prepare_mlrc_task_ledger(
        task_name="product-recommendation",
        ledger_dir=ledger_dir,
        candidate_specs_path=specs_path,
        labels_path=labels_path,
        incumbent_id="baseline",
    )

    assert prepared.candidate_count == 2
    assert prepared.unit_count == 4
    candidate_rows = read_jsonl(ledger_dir / "candidates.jsonl")
    assert candidate_rows[0]["score"] == pytest.approx((0.5 + 1 / 3 + 0.0 + 1.0) / 4)
    assert candidate_rows[1]["score"] == pytest.approx((1.0 + 1.0 + 1.0 + 0.5) / 4)
    assert candidate_rows[1]["unit_outputs_status"] == "per_unit_bootstrap_ok"


def test_product_recommendation_cli_prepares_and_runs_per_unit_detector(tmp_path):
    labels_path = tmp_path / "labels.csv"
    baseline_path = tmp_path / "baseline_pred.csv"
    candidate_path = tmp_path / "candidate_pred.csv"
    specs_path = tmp_path / "candidate_specs.jsonl"
    ledger_dir = tmp_path / "ledger"

    _write_csv(
        labels_path,
        [
            {"session_id": f"s{i}", "next_item": target}
            for i, target in enumerate(["a", "b", "c", "d", "e", "f"], start=1)
        ],
    )
    _write_csv(
        baseline_path,
        [
            {"session_id": "s1", "locale": "US", "next_item_prediction": "['x', 'a']"},
            {"session_id": "s2", "locale": "US", "next_item_prediction": "['x', 'b']"},
            {"session_id": "s3", "locale": "US", "next_item_prediction": "['x', 'c']"},
            {"session_id": "s4", "locale": "US", "next_item_prediction": "['q']"},
            {"session_id": "s5", "locale": "US", "next_item_prediction": "['q']"},
            {"session_id": "s6", "locale": "US", "next_item_prediction": "['q']"},
        ],
    )
    _write_csv(
        candidate_path,
        [
            {"session_id": "s1", "locale": "US", "next_item_prediction": "['a']"},
            {"session_id": "s2", "locale": "US", "next_item_prediction": "['b']"},
            {"session_id": "s3", "locale": "US", "next_item_prediction": "['c']"},
            {"session_id": "s4", "locale": "US", "next_item_prediction": "['d']"},
            {"session_id": "s5", "locale": "US", "next_item_prediction": "['e']"},
            {"session_id": "s6", "locale": "US", "next_item_prediction": "['f']"},
        ],
    )
    write_jsonl(
        specs_path,
        [
            {"candidate_id": "baseline", "prediction_path": str(baseline_path), "is_incumbent": True},
            {"candidate_id": "cand_better", "prediction_path": str(candidate_path)},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_verifier.py",
            "--ledger-dir",
            str(ledger_dir),
            "--mlrc-task",
            "product-recommendation",
            "--candidate-specs",
            str(specs_path),
            "--labels-path",
            str(labels_path),
            "--bootstrap-samples",
            "200",
            "--epsilon",
            "0.05",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads((ledger_dir / "verifier_report.json").read_text())
    manifest = json.loads((ledger_dir / "mlrc_task_manifest.json").read_text())
    assert manifest["task_name"] == "product-recommendation"
    assert report["detector_mode"] == "per_unit_bootstrap_ok"
    assert report["candidates"][0]["candidate_id"] == "cand_better"
    assert read_jsonl(ledger_dir / "unit_outputs" / "cand_better.jsonl")[0]["unit_type"] == "session"


def test_mlrc_workflow_selects_best_dev_and_writes_specs(tmp_path):
    idea_evals_path = tmp_path / "idea_evals.json"
    baseline_path = tmp_path / "preds" / "baseline.csv"
    candidate_path = tmp_path / "preds" / "candidate.csv"
    specs_path = tmp_path / "candidate_specs.jsonl"

    write_json(
        idea_evals_path,
        {
            "implementations": [
                {"step": 2, "method_name": "weak_method", "phase": "dev", "performance": 0.2},
                {"step": 4, "method_name": "best_method", "phase": "dev", "performance": 0.4},
                {"step": 5, "method_name": "test_method", "phase": "test", "performance": 0.5},
            ]
        },
    )
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text("next_item_prediction\n['a']\n", encoding="utf-8")
    candidate_path.write_text("next_item_prediction\n['b']\n", encoding="utf-8")

    best = select_best_dev_implementation(idea_evals_path)
    assert best.step == 4
    assert best.method_name == "best_method"
    assert best.performance == pytest.approx(0.4)

    write_product_candidate_specs(
        specs_path,
        baseline_prediction_path=baseline_path,
        candidate_prediction_path=candidate_path,
        candidate_method_name=best.method_name,
        candidate_step=best.step,
        run_id="run_001",
    )
    rows = read_jsonl(specs_path)
    assert rows[0]["candidate_id"] == "baseline"
    assert rows[0]["is_incumbent"] is True
    assert rows[1]["candidate_id"] == "best_dev"
    assert rows[1]["method_name"] == "best_method"
    assert rows[1]["step"] == 4


def test_mlrc_verifier_hook_cli_writes_action_and_feedback(tmp_path):
    labels_path = tmp_path / "labels.csv"
    baseline_path = tmp_path / "baseline_pred.csv"
    candidate_path = tmp_path / "candidate_pred.csv"
    ledger_dir = tmp_path / "ledger"

    _write_csv(
        labels_path,
        [
            {"session_id": f"s{i}", "next_item": target}
            for i, target in enumerate(["a", "b", "c", "d", "e", "f"], start=1)
        ],
    )
    _write_csv(
        baseline_path,
        [
            {"session_id": "s1", "locale": "US", "next_item_prediction": "['x', 'a']"},
            {"session_id": "s2", "locale": "US", "next_item_prediction": "['x', 'b']"},
            {"session_id": "s3", "locale": "US", "next_item_prediction": "['x', 'c']"},
            {"session_id": "s4", "locale": "US", "next_item_prediction": "['q']"},
            {"session_id": "s5", "locale": "US", "next_item_prediction": "['q']"},
            {"session_id": "s6", "locale": "US", "next_item_prediction": "['q']"},
        ],
    )
    _write_csv(
        candidate_path,
        [
            {"session_id": "s1", "locale": "US", "next_item_prediction": "['a']"},
            {"session_id": "s2", "locale": "US", "next_item_prediction": "['b']"},
            {"session_id": "s3", "locale": "US", "next_item_prediction": "['c']"},
            {"session_id": "s4", "locale": "US", "next_item_prediction": "['d']"},
            {"session_id": "s5", "locale": "US", "next_item_prediction": "['e']"},
            {"session_id": "s6", "locale": "US", "next_item_prediction": "['f']"},
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/mlrc_verifier_hook.py",
            "--ledger-dir",
            str(ledger_dir),
            "--baseline-pred",
            str(baseline_path),
            "--candidate-pred",
            str(candidate_path),
            "--labels-path",
            str(labels_path),
            "--candidate-id",
            "cand_better",
            "--candidate-method",
            "better_method",
            "--bootstrap-samples",
            "50",
            "--epsilon",
            "0.05",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads((ledger_dir / "hook_summary.json").read_text())
    feedback = (ledger_dir / "agent_feedback.txt").read_text()
    assert summary["recommended_action"]["candidate_id"] == "cand_better"
    assert summary["scheduler_next_action"]["type"] in {"final_audit", "stop_unresolved", "rerun_candidate"}
    assert "Verifier decision:" in feedback
    assert "cand_better" in feedback


@pytest.mark.parametrize(
    ("action_type", "expected_text"),
    [
        ("final_audit", "clears the verifier evidence gate"),
        ("rerun_candidate", "gain looks positive but unresolved"),
        ("stop_unresolved", "Do not finalize this candidate"),
    ],
)
def test_agent_feedback_includes_action_specific_steering(action_type, expected_text):
    feedback = build_agent_feedback(
        {
            "recommended_action": {
                "type": action_type,
                "candidate_id": "cand_001",
                "reason": "test reason",
            },
            "candidates": [
                {
                    "candidate_id": "cand_001",
                    "raw_delta": 0.02,
                    "paired_delta": 0.021,
                    "paired_se": 0.003,
                    "dev_resolution_ratio": 4.0,
                    "decision": action_type,
                }
            ],
        },
        {"type": action_type, "candidate_id": "cand_001", "reason": "test reason"},
    )

    assert "- steering:" in feedback
    assert expected_text in feedback
    assert "Use this as verifier evidence; do not treat it as a new benchmark score." in feedback


def test_mlrc_veritas_hook_patcher_handles_clean_low_level_actions(tmp_path):
    target = tmp_path / "low_level_actions.py"
    target.write_text(_low_level_actions_fixture(manual_pythonpath=False), encoding="utf-8")

    assert patch_low_level_actions(target) is True
    patched = target.read_text(encoding="utf-8")
    assert "_maybe_run_veritas_hook" in patched
    assert 'VERITAS_HOOK_PATCH_VERSION = "2_steering"' in patched
    assert "VERITAS_HOOK_ENABLE" in patched
    assert "VERITAS_HOOK_STEER" in patched
    assert "agent_feedback.txt" in patched
    assert 'os.getenv("VERITAS_HOOK_EPSILON", "0.01")' in patched
    assert "repo_root = os.path.abspath" in patched
    assert "veritas_observation = _maybe_run_veritas_hook" in patched
    assert patch_low_level_actions(target) is False


def test_mlrc_veritas_hook_patcher_handles_manual_pythonpath_fix(tmp_path):
    target = tmp_path / "low_level_actions.py"
    target.write_text(_low_level_actions_fixture(manual_pythonpath=True), encoding="utf-8")

    assert patch_low_level_actions(target) is True
    patched = target.read_text(encoding="utf-8")
    assert "_maybe_run_veritas_hook" in patched
    assert patched.count("repo_root = os.path.abspath") == 1
    assert "PYTHONPATH={repo_root}:`pwd`" in patched


def test_mlrc_veritas_hook_patcher_upgrades_passive_hook(tmp_path):
    target = tmp_path / "low_level_actions.py"
    target.write_text(_low_level_actions_fixture(manual_pythonpath=False), encoding="utf-8")

    passive = patch_low_level_actions(target)
    assert passive is True

    old_style = target.read_text(encoding="utf-8").replace(
        'VERITAS_HOOK_PATCH_VERSION = "2_steering"\n\n\n',
        "",
    ).replace(
        "\n\ndef _veritas_hook_steering_enabled():\n"
        '    return os.getenv("VERITAS_HOOK_STEER", "").strip().lower() in {"1", "true", "yes", "on"}\n',
        "",
    ).replace(
        '\n    if _veritas_hook_steering_enabled():\n'
        '        feedback = _read_veritas_agent_feedback(hook_ledger_dir)\n'
        '        if feedback:\n'
        '            return feedback\n'
        '        return f"VeriTAS hook: verifier succeeded but no agent feedback was found at {hook_ledger_dir}"\n',
        "\n",
    )
    target.write_text(old_style, encoding="utf-8")

    assert patch_low_level_actions(target) is True
    upgraded = target.read_text(encoding="utf-8")
    assert 'VERITAS_HOOK_PATCH_VERSION = "2_steering"' in upgraded
    assert "VERITAS_HOOK_STEER" in upgraded
    assert "agent_feedback.txt" in upgraded


def test_mlrc_task_adapter_rejects_unsupported_task(tmp_path):
    specs_path = tmp_path / "candidate_specs.jsonl"
    labels_path = tmp_path / "labels.csv"
    write_jsonl(specs_path, [{"candidate_id": "baseline", "prediction_path": "pred.csv"}])
    _write_csv(labels_path, [{"next_item": "a"}])

    with pytest.raises(MlrcTaskAdapterError, match="unsupported MLRC task"):
        prepare_mlrc_task_ledger(
            task_name="weather_forcast",
            ledger_dir=tmp_path / "ledger",
            candidate_specs_path=specs_path,
            labels_path=labels_path,
        )


def _candidate(
    candidate_id: str,
    score: float,
    *,
    score_direction: str = "maximize",
    unit_outputs_status: UnitOutputStatus = UnitOutputStatus.SINGLE_SCALAR_NO_SE,
    validity_flags: list[str] | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        logical_candidate_id=candidate_id,
        run_id="run_001",
        card_id=None,
        parent_candidate_id="baseline" if candidate_id != "baseline" else None,
        comparison_origin_id="baseline",
        pool_id="pool_001",
        step=None,
        method_name=candidate_id,
        phase="dev",
        seed=0,
        score=score,
        score_direction=score_direction,
        score_source="output/idea_evals.json",
        score_extracted_by="runner_wrapper",
        llm_reported_score=None,
        llm_score_trusted=False,
        snapshot_path=f"snapshots/{candidate_id}",
        unit_outputs_status=unit_outputs_status,
        validity_flags=validity_flags or [],
    )


def _scheduler_state(queue_threshold: int = 0) -> dict:
    return {
        "budget": {
            "max_dev_runs": 10,
            "used_dev_runs": 2,
            "max_reruns_per_candidate": 3,
            "max_audited_candidates": 1,
            "used_audited_candidates": 0,
        },
        "queue_threshold": queue_threshold,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _low_level_actions_fixture(*, manual_pythonpath: bool) -> str:
    if manual_pythonpath:
        command_block = (
            '        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))\n'
            '        cmd = f"PYTHONPATH={repo_root}:`pwd`:${PYTHONPATH:-} CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"\n'
        )
    else:
        command_block = (
            '        cmd = f"PYTHONPATH=`pwd` CUDA_VISIBLE_DEVICES={device} {python} -u {script_path}"\n'
        )
    return (
        '"""fixture"""\n'
        "import os\n"
        "import json\n"
        "import subprocess\n"
        "import selectors\n"
        "import shutil\n"
        "import glob\n"
        "import sys\n"
        "import inspect\n"
        "\n"
        "def safe_copy_file(src, dst):\n"
        "    shutil.copyfile(src, dst)\n"
        "\n\n"
        '# @check_file_in_work_dir(["script_name_and_args"])\n'
        "def execute_script(script_name_and_args, work_dir = \".\", **kwargs):\n"
        "    try:\n"
        "        script_path = script_name_and_args\n"
        "        device = kwargs[\"device\"]\n"
        "        python = kwargs[\"python\"]\n"
        f"{command_block}"
        "        return_code = 0\n"
        "        observation = \"\"\n"
        "        stderr_lines = []\n"
        "        if observation == \"\" and return_code == 0:\n"
        "            # printed to stderr only\n"
        "            observation = \"\".join(stderr_lines)\n"
        "        return \"The script has been executed. Here is the output:\\n\" + observation\n"
        "    except Exception:\n"
        "        raise\n"
    )
