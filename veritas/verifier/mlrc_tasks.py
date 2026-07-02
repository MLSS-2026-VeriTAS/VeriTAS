"""MLRC task-specific adapters for emitting verifier unit outputs."""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veritas.verifier.enums import UnitOutputStatus
from veritas.verifier.io import read_jsonl, write_json, write_jsonl
from veritas.verifier.types import CandidateRecord


SUPPORTED_TASKS = ("product-recommendation",)
PRODUCT_RECOMMENDATION_METRICS = ("parsed_mrr", "mlrc_exact")


class MlrcTaskAdapterError(ValueError):
    """Raised when MLRC task artifacts cannot be converted to verifier inputs."""


@dataclass
class PreparedTaskLedger:
    task_name: str
    ledger_dir: Path
    candidate_count: int
    unit_count: int
    metric_variant: str


def prepare_mlrc_task_ledger(
    *,
    task_name: str,
    ledger_dir: str | Path,
    candidate_specs_path: str | Path,
    labels_path: str | Path,
    phase: str = "dev",
    metric_variant: str = "parsed_mrr",
    incumbent_id: str = "baseline",
    score_recompute_tolerance: float = 1e-9,
) -> PreparedTaskLedger:
    """Prepare verifier ledger artifacts for a supported MLRC task."""

    if task_name != "product-recommendation":
        raise MlrcTaskAdapterError(
            f"unsupported MLRC task {task_name!r}; supported tasks: {', '.join(SUPPORTED_TASKS)}"
        )
    if metric_variant not in PRODUCT_RECOMMENDATION_METRICS:
        raise MlrcTaskAdapterError(
            f"unsupported product-recommendation metric variant {metric_variant!r}"
        )

    ledger_path = Path(ledger_dir)
    specs = _load_candidate_specs(candidate_specs_path)
    labels = _read_csv_rows(labels_path)
    candidates: list[CandidateRecord] = []
    unit_count: int | None = None

    for spec in specs:
        candidate = _prepare_product_recommendation_candidate(
            ledger_path=ledger_path,
            spec=spec,
            labels=labels,
            phase=phase,
            metric_variant=metric_variant,
            incumbent_id=incumbent_id,
            score_recompute_tolerance=score_recompute_tolerance,
        )
        rows = read_jsonl(ledger_path / str(candidate.unit_outputs_path))
        if unit_count is None:
            unit_count = len(rows)
        elif unit_count != len(rows):
            raise MlrcTaskAdapterError("candidate unit output counts do not align")
        candidates.append(candidate)

    if not any(candidate.candidate_id == incumbent_id for candidate in candidates):
        raise MlrcTaskAdapterError(f"candidate specs must include incumbent_id={incumbent_id!r}")

    write_jsonl(ledger_path / "candidates.jsonl", candidates)
    manifest = {
        "task_name": task_name,
        "phase": phase,
        "metric_variant": metric_variant,
        "incumbent_id": incumbent_id,
        "candidate_count": len(candidates),
        "unit_count": unit_count or 0,
        "candidate_specs_path": str(candidate_specs_path),
        "labels_path": str(labels_path),
    }
    write_json(ledger_path / "mlrc_task_manifest.json", manifest)

    return PreparedTaskLedger(
        task_name=task_name,
        ledger_dir=ledger_path,
        candidate_count=len(candidates),
        unit_count=unit_count or 0,
        metric_variant=metric_variant,
    )


def _load_candidate_specs(path: str | Path) -> list[dict[str, Any]]:
    specs = read_jsonl(path)
    if not specs:
        raise MlrcTaskAdapterError("candidate specs JSONL is empty")
    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise MlrcTaskAdapterError(f"candidate spec {index} is not an object")
        if not spec.get("prediction_path"):
            raise MlrcTaskAdapterError(f"candidate spec {index} has no prediction_path")
    return specs


def _prepare_product_recommendation_candidate(
    *,
    ledger_path: Path,
    spec: dict[str, Any],
    labels: list[dict[str, str]],
    phase: str,
    metric_variant: str,
    incumbent_id: str,
    score_recompute_tolerance: float,
) -> CandidateRecord:
    candidate_id = str(spec.get("candidate_id") or spec.get("method_name") or "candidate")
    method_name = str(spec.get("method_name") or candidate_id)
    predictions = _read_csv_rows(spec["prediction_path"])
    unit_rows = product_recommendation_unit_outputs(
        predictions=predictions,
        labels=labels,
        candidate_id=candidate_id,
        phase=str(spec.get("phase") or phase),
        metric_variant=metric_variant,
    )
    score = _mean(row["score_component"] for row in unit_rows)
    expected_score = spec.get("score")
    if expected_score is not None and abs(score - float(expected_score)) > score_recompute_tolerance:
        raise MlrcTaskAdapterError(
            f"candidate {candidate_id} score mismatch: recomputed={score}, expected={expected_score}"
        )

    unit_outputs_path = Path("unit_outputs") / f"{_safe_file_name(candidate_id)}.jsonl"
    write_jsonl(ledger_path / unit_outputs_path, unit_rows)

    is_incumbent = candidate_id == incumbent_id or bool(spec.get("is_incumbent", False))
    return CandidateRecord(
        candidate_id=candidate_id,
        logical_candidate_id=str(spec.get("logical_candidate_id") or candidate_id),
        run_id=str(spec.get("run_id") or f"{candidate_id}_run"),
        card_id=spec.get("card_id"),
        parent_candidate_id=None if is_incumbent else str(spec.get("parent_candidate_id") or incumbent_id),
        comparison_origin_id=str(spec.get("comparison_origin_id") or incumbent_id),
        pool_id=str(spec.get("pool_id") or "mlrc_product_recommendation_dev"),
        step=_optional_int(spec.get("step")),
        method_name=method_name,
        phase=str(spec.get("phase") or phase),
        seed=_optional_int(spec.get("seed")),
        score=score,
        score_direction="maximize",
        score_source=str(spec["prediction_path"]),
        score_extracted_by="mlrc_product_recommendation_adapter",
        llm_reported_score=None,
        llm_score_trusted=False,
        snapshot_path=str(spec.get("snapshot_path") or Path("snapshots") / candidate_id),
        snapshot_source_path=spec.get("snapshot_source_path"),
        unit_outputs_status=UnitOutputStatus.PER_UNIT_BOOTSTRAP_OK,
        unit_outputs_path=str(unit_outputs_path),
        verifier_only_unit_scores_path=str(unit_outputs_path),
        metric_recomputed_score=score,
        score_recompute_abs_error=0.0,
        score_recompute_tolerance=score_recompute_tolerance,
        code_diff_path=spec.get("code_diff_path"),
        config_path=spec.get("config_path"),
        validity_flags=[],
    )


def product_recommendation_unit_outputs(
    *,
    predictions: list[dict[str, str]],
    labels: list[dict[str, str]],
    candidate_id: str,
    phase: str,
    metric_variant: str,
) -> list[dict[str, Any]]:
    if len(predictions) != len(labels):
        raise MlrcTaskAdapterError(
            f"prediction/label row count mismatch: {len(predictions)} != {len(labels)}"
        )
    if metric_variant not in PRODUCT_RECOMMENDATION_METRICS:
        raise MlrcTaskAdapterError(f"unsupported metric variant {metric_variant!r}")

    rows: list[dict[str, Any]] = []
    for index, (prediction_row, label_row) in enumerate(zip(predictions, labels)):
        if "next_item_prediction" not in prediction_row:
            raise MlrcTaskAdapterError("prediction CSV must contain next_item_prediction")
        if "next_item" not in label_row:
            raise MlrcTaskAdapterError("labels CSV must contain next_item")

        raw_prediction = prediction_row["next_item_prediction"]
        target = str(label_row["next_item"])
        parsed_predictions = _parse_prediction_list(raw_prediction)
        if metric_variant == "mlrc_exact":
            score_component, rank = _mlrc_exact_reciprocal_rank(raw_prediction, target)
        else:
            score_component, rank = _parsed_reciprocal_rank(parsed_predictions, target)

        unit_id = _unit_id(index, prediction_row, label_row)
        metadata = {
            "candidate_id": candidate_id,
            "task_name": "product-recommendation",
            "metric_variant": metric_variant,
            "rank": rank,
            "locale": prediction_row.get("locale") or label_row.get("locale"),
        }
        rows.append(
            {
                "unit_id": unit_id,
                "unit_type": "session",
                "phase": phase,
                "prediction": parsed_predictions,
                "target_or_verifier_label": target,
                "score_component": score_component,
                "weight": 1.0,
                "metadata": metadata,
            }
        )
    return rows


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {str(key): "" if value is None else value for key, value in row.items()}
            for row in reader
        ]


def _parse_prediction_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]

    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    if isinstance(parsed, str):
        return [parsed]

    stripped = text.strip("[]").replace("\n", " ").replace("\r", " ")
    if "," in stripped:
        pieces = stripped.split(",")
    else:
        pieces = stripped.split()
    return [
        piece.strip().strip("'\"")
        for piece in pieces
        if piece.strip().strip("'\"")
    ]


def _parsed_reciprocal_rank(predictions: list[str], target: str) -> tuple[float, int | None]:
    try:
        rank = predictions.index(target) + 1
    except ValueError:
        return 0.0, None
    return 1.0 / rank, rank


def _mlrc_exact_reciprocal_rank(raw_prediction: str, target: str) -> tuple[float, int | None]:
    try:
        rank = str(raw_prediction).index(target) + 1
    except ValueError:
        return 0.0, None
    return 1.0 / rank, rank


def _unit_id(index: int, prediction_row: dict[str, str], label_row: dict[str, str]) -> str:
    for key in ("session_id", "id"):
        if prediction_row.get(key):
            return str(prediction_row[key])
        if label_row.get(key):
            return str(label_row[key])
    return f"row_{index:06d}"


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        raise MlrcTaskAdapterError("cannot score an empty candidate")
    return sum(items) / len(items)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_file_name(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("._")
    return safe or "candidate"
