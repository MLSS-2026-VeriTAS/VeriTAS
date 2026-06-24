"""Paired bootstrap over aligned per-unit outputs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from veritas.verifier.types import BootstrapResult
from veritas.verifier.unit_outputs import index_unit_outputs, recompute_mean_score


def paired_bootstrap(
    *,
    incumbent_rows: list[dict[str, Any]],
    candidate_rows_by_id: dict[str, list[dict[str, Any]]],
    score_direction: str,
    samples: int = 1000,
    seed: int = 0,
    group_field: str | None = None,
) -> BootstrapResult:
    if samples < 2:
        raise ValueError("paired bootstrap requires at least two samples")
    if not candidate_rows_by_id:
        raise ValueError("paired bootstrap requires at least one candidate")
    if score_direction not in {"maximize", "minimize"}:
        raise ValueError(f"invalid score direction: {score_direction}")

    direction_multiplier = 1.0 if score_direction == "maximize" else -1.0

    incumbent_by_id = index_unit_outputs(incumbent_rows)
    candidate_by_id = {
        candidate_id: index_unit_outputs(rows)
        for candidate_id, rows in candidate_rows_by_id.items()
    }
    unit_ids = sorted(incumbent_by_id)

    for candidate_id, rows_by_id in candidate_by_id.items():
        if set(rows_by_id) != set(unit_ids):
            raise ValueError(f"candidate {candidate_id} unit IDs do not align with incumbent")

    sample_units = _sample_units_by_group(unit_ids, incumbent_by_id, group_field)
    rng = random.Random(seed)
    candidate_order = sorted(candidate_rows_by_id)
    bootstrap_deltas: list[list[float]] = []

    for _ in range(samples):
        sampled_units = _draw_units(sample_units, rng)
        incumbent_sample = [incumbent_by_id[unit_id] for unit_id in sampled_units]
        incumbent_score = recompute_mean_score(incumbent_sample)

        row: list[float] = []
        for candidate_id in candidate_order:
            candidate_sample = [
                candidate_by_id[candidate_id][unit_id]
                for unit_id in sampled_units
            ]
            raw_delta = recompute_mean_score(candidate_sample) - incumbent_score
            row.append(direction_multiplier * raw_delta)
        bootstrap_deltas.append(row)

    means = _column_means(bootstrap_deltas)
    covariance = _sample_covariance(bootstrap_deltas, means)
    paired_se = {
        candidate_id: math.sqrt(max(covariance[index][index], 0.0))
        for index, candidate_id in enumerate(candidate_order)
    }
    delta_hat = {
        candidate_id: means[index]
        for index, candidate_id in enumerate(candidate_order)
    }
    return BootstrapResult(
        candidate_order=candidate_order,
        delta_hat=delta_hat,
        paired_se=paired_se,
        covariance=covariance,
    )


def _sample_units_by_group(
    unit_ids: list[str],
    incumbent_by_id: dict[str, dict[str, Any]],
    group_field: str | None,
) -> list[list[str]]:
    if group_field is None:
        return [[unit_id] for unit_id in unit_ids]

    grouped: dict[str, list[str]] = defaultdict(list)
    for unit_id in unit_ids:
        group_id = incumbent_by_id[unit_id].get(group_field)
        if group_id is None:
            raise ValueError(f"unit {unit_id} has no group field {group_field}")
        grouped[str(group_id)].append(unit_id)
    return [sorted(values) for _, values in sorted(grouped.items())]


def _draw_units(sample_units: list[list[str]], rng: random.Random) -> list[str]:
    drawn: list[str] = []
    for _ in range(len(sample_units)):
        drawn.extend(rng.choice(sample_units))
    return drawn


def _column_means(rows: list[list[float]]) -> list[float]:
    width = len(rows[0])
    return [
        sum(row[column] for row in rows) / len(rows)
        for column in range(width)
    ]


def _sample_covariance(rows: list[list[float]], means: list[float]) -> list[list[float]]:
    width = len(means)
    denom = len(rows) - 1
    covariance = [[0.0 for _ in range(width)] for _ in range(width)]
    for row in rows:
        for i in range(width):
            for j in range(width):
                covariance[i][j] += (row[i] - means[i]) * (row[j] - means[j])
    return [
        [value / denom for value in covariance_row]
        for covariance_row in covariance
    ]
