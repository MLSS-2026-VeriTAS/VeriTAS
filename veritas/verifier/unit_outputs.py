"""Per-unit output loading, alignment, and metric recomputation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from veritas.verifier.enums import UnitOutputStatus
from veritas.verifier.io import read_jsonl


class UnitOutputError(ValueError):
    """Raised when per-unit output artifacts are invalid."""


def load_unit_outputs(path: str | Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise UnitOutputError(f"unit output row {index} is not an object")
        if not row.get("unit_id"):
            raise UnitOutputError(f"unit output row {index} has no unit_id")
    return rows


def index_unit_outputs(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        unit_id = str(row["unit_id"])
        if unit_id in indexed:
            raise UnitOutputError(f"duplicate unit_id: {unit_id}")
        indexed[unit_id] = row
    return indexed


def align_unit_outputs(
    incumbent_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    incumbent_by_id = index_unit_outputs(incumbent_rows)
    candidate_by_id = index_unit_outputs(candidate_rows)

    if set(incumbent_by_id) != set(candidate_by_id):
        missing_from_candidate = sorted(set(incumbent_by_id) - set(candidate_by_id))
        missing_from_incumbent = sorted(set(candidate_by_id) - set(incumbent_by_id))
        raise UnitOutputError(
            "unit_id mismatch: "
            f"missing_from_candidate={missing_from_candidate}, "
            f"missing_from_incumbent={missing_from_incumbent}"
        )

    return [
        (incumbent_by_id[unit_id], candidate_by_id[unit_id])
        for unit_id in sorted(incumbent_by_id)
    ]


def score_component(row: dict[str, Any]) -> float:
    component = row.get("score_component")
    if component is not None:
        return float(component)

    if "prediction" in row and "target_or_verifier_label" in row:
        return 1.0 if row["prediction"] == row["target_or_verifier_label"] else 0.0

    raise UnitOutputError(f"cannot derive score component for unit_id={row.get('unit_id')}")


def recompute_mean_score(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise UnitOutputError("cannot recompute score from empty unit outputs")

    weighted_sum = 0.0
    total_weight = 0.0
    for row in rows:
        weight = row.get("weight", 1.0)
        weight = 1.0 if weight is None else float(weight)
        weighted_sum += score_component(row) * weight
        total_weight += weight

    if total_weight <= 0:
        raise UnitOutputError("total unit weight must be positive")
    return weighted_sum / total_weight


def recompute_score_check(
    rows: list[dict[str, Any]],
    expected_score: float,
    tolerance: float,
) -> tuple[float, float, UnitOutputStatus]:
    recomputed = recompute_mean_score(rows)
    abs_error = abs(recomputed - expected_score)
    status = (
        UnitOutputStatus.PER_UNIT_BOOTSTRAP_OK
        if abs_error <= tolerance
        else UnitOutputStatus.SCORE_RECOMPUTE_MISMATCH
    )
    return recomputed, abs_error, status
