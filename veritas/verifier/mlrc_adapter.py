"""Utilities for parsing MLRC-Bench scalar score artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from veritas.verifier.enums import UnitOutputStatus
from veritas.verifier.io import read_json
from veritas.verifier.types import CandidateRecord


class MlrcArtifactError(ValueError):
    """Raised when an MLRC artifact cannot be parsed into candidate records."""


def load_idea_evals(path: str | Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("implementations"), list):
        implementations = data["implementations"]
    elif isinstance(data, list):
        implementations = data
    else:
        raise MlrcArtifactError("idea_evals artifact must contain an implementations list")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(implementations):
        if not isinstance(item, dict):
            raise MlrcArtifactError(f"implementation at index {index} is not an object")
        records.append(item)
    return records


def candidates_from_idea_evals(
    path: str | Path,
    *,
    run_id: str,
    card_id: str | None,
    comparison_origin_id: str,
    pool_id: str,
    phase: str = "dev",
    seed: int | None = None,
    parent_candidate_id: str | None = "baseline",
    score_direction: str = "maximize",
    log_dir: str | None = None,
    score_source: str = "output/idea_evals.json",
) -> list[CandidateRecord]:
    implementations = load_idea_evals(path)
    candidates: list[CandidateRecord] = []

    for index, implementation in enumerate(implementations):
        score = _extract_score(implementation, index)
        method_name = str(implementation.get("method_name") or f"candidate_{index:03d}")
        step = _optional_int(implementation.get("step"))
        candidate_id = _candidate_id(run_id, method_name, step, index)
        snapshot_source_path = _snapshot_source_path(log_dir, step)

        candidates.append(
            CandidateRecord(
                candidate_id=candidate_id,
                logical_candidate_id=_slug(method_name),
                run_id=run_id,
                card_id=card_id,
                parent_candidate_id=parent_candidate_id,
                comparison_origin_id=comparison_origin_id,
                pool_id=pool_id,
                step=step,
                method_name=method_name,
                phase=phase,
                seed=seed,
                score=score,
                score_direction=score_direction,
                score_source=score_source,
                score_extracted_by="runner_wrapper",
                llm_reported_score=None,
                llm_score_trusted=False,
                snapshot_source_path=snapshot_source_path,
                snapshot_path=_copied_snapshot_path(candidate_id, step),
                unit_outputs_status=UnitOutputStatus.SINGLE_SCALAR_NO_SE,
                validity_flags=[UnitOutputStatus.SINGLE_SCALAR_NO_SE.value],
            )
        )

    return candidates


def _extract_score(implementation: dict[str, Any], index: int) -> float:
    if "performance" not in implementation:
        raise MlrcArtifactError(f"implementation at index {index} has no performance field")
    score = implementation["performance"]
    if score is None:
        raise MlrcArtifactError(f"implementation at index {index} has null performance")
    try:
        return float(score)
    except (TypeError, ValueError) as exc:
        raise MlrcArtifactError(
            f"implementation at index {index} has non-numeric performance"
        ) from exc


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "candidate"


def _candidate_id(run_id: str, method_name: str, step: int | None, index: int) -> str:
    step_part = f"step_{step}" if step is not None else f"idx_{index}"
    return f"{_slug(run_id)}_{step_part}_{_slug(method_name)}"


def _snapshot_source_path(log_dir: str | None, step: int | None) -> str | None:
    if not log_dir or step is None:
        return None
    return str(Path(log_dir) / "env_log" / "traces" / f"step_{step}_files")


def _copied_snapshot_path(candidate_id: str, step: int | None) -> str:
    suffix = f"step_{step}_files" if step is not None else "snapshot"
    return str(Path("snapshots") / f"{candidate_id}_{suffix}")

