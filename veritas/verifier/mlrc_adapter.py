"""Adapters from MLRC-Bench logs to Verifier inputs.

These functions translate the artifacts that MLRC-Bench already produces into
:class:`IterationObservation` objects, so the Verifier can be attached to a real
run later with no changes to its core. They are deliberately pure (dict in,
objects out) so they can be unit-tested on small synthetic payloads without any
benchmark, GPU, or API access. Thin ``*_from_file`` helpers load JSON from disk.

Reference shapes (see ``MLAgentBench/eval.py`` and
``MLAgentBench/LLM_as_a_Judge.py`` on the ``copyMLRC`` branch):

* ``EvaluationResult`` carries ``score`` (list of per-step scores), the matching
  ``score_steps`` (step indices), and a scalar ``final_score``.
* A results file maps ``trace.json`` path -> serialized ``EvaluationResult``.
* The LLM judge returns ``{"with_code"|"without_code": {metric: {"Rating": int,
  ...}}}``.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .types import IterationObservation

# Rubric dimensions emitted by MLAgentBench/LLM_as_a_Judge.py.
RUBRIC_DIMENSIONS = (
    "Clarity",
    "Validity",
    "Rigorousness",
    "Innovativeness",
    "Generalizability",
)


def rubric_ratings_from_judge(
    judge_output: Dict[str, object], use_code: bool = False
) -> Optional[Dict[str, float]]:
    """Extract ``{dimension: rating}`` from an ``llm_evaluate_method`` result.

    Args:
        judge_output: The dict returned by ``LLM_as_a_Judge.llm_evaluate_method``
            with ``"with_code"`` and ``"without_code"`` sections.
        use_code: Select the code-aware ratings when ``True``.

    Returns:
        A mapping from rubric dimension to integer rating, or ``None`` if no
        usable ratings are present.
    """
    section_key = "with_code" if use_code else "without_code"
    section = judge_output.get(section_key) if isinstance(judge_output, dict) else None
    if not isinstance(section, dict):
        return None

    ratings: Dict[str, float] = {}
    for dim, payload in section.items():
        if isinstance(payload, dict) and "Rating" in payload and payload["Rating"] is not None:
            try:
                ratings[dim] = float(payload["Rating"])
            except (TypeError, ValueError):
                continue
    return ratings or None


def observations_from_evaluation_result(
    result: Dict[str, object],
    higher_is_better: bool = True,
    include_final: bool = True,
    rubric_by_step: Optional[Dict[int, Dict[str, float]]] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> List[IterationObservation]:
    """Build per-iteration observations from one serialized ``EvaluationResult``.

    Each evaluated step becomes one :class:`IterationObservation` carrying that
    step's single objective score. When ``score_steps`` is shorter than
    ``score`` and ``include_final`` is set, the trailing score is treated as the
    final-answer iteration.
    """
    base_meta = dict(metadata or {})
    scores = list(result.get("score", []) or [])  # type: ignore[arg-type]
    steps = list(result.get("score_steps", []) or [])  # type: ignore[arg-type]

    observations: List[IterationObservation] = []
    for idx, step in enumerate(steps):
        if idx >= len(scores):
            break
        step_int = int(step)
        meta = dict(base_meta)
        meta["step"] = step_int
        observations.append(
            IterationObservation(
                iteration=idx,
                scores=[float(scores[idx])],
                higher_is_better=higher_is_better,
                rubric_ratings=(rubric_by_step or {}).get(step_int),
                metadata=meta,
            )
        )

    # Trailing final score not covered by score_steps.
    if include_final and len(scores) > len(steps):
        meta = dict(base_meta)
        meta["step"] = "final"
        observations.append(
            IterationObservation(
                iteration=len(observations),
                scores=[float(scores[-1])],
                higher_is_better=higher_is_better,
                metadata=meta,
            )
        )

    return observations


def observations_from_results(
    results: Dict[str, Dict[str, object]],
    higher_is_better: bool = True,
) -> Dict[str, List[IterationObservation]]:
    """Convert a full results mapping (path -> EvaluationResult) into observations."""
    out: Dict[str, List[IterationObservation]] = {}
    for path, result in results.items():
        out[path] = observations_from_evaluation_result(
            result, higher_is_better=higher_is_better, metadata={"path": path}
        )
    return out


def merge_seed_observations(
    runs: List[List[IterationObservation]],
    higher_is_better: bool = True,
) -> List[IterationObservation]:
    """Combine matched iterations across repeated seeds into multi-score steps.

    Given several runs that share the same iteration structure (for example the
    same task evaluated under different seeds), produce one observation per
    iteration whose ``scores`` gathers every seed's measurement. This lets the
    Verifier infer the noise variance directly from seed spread.
    """
    if not runs:
        return []
    reference = runs[0]

    def obs_key(obs: IterationObservation):
        has_step = "step" in obs.metadata
        step_value = obs.metadata.get("step") if has_step else None
        return obs.iteration, has_step, step_value

    expected_keys = [obs_key(obs) for obs in reference]
    for idx, run in enumerate(runs[1:], start=1):
        run_keys = [obs_key(obs) for obs in run]
        if run_keys != expected_keys:
            raise ValueError(
                "merge_seed_observations requires all runs to have matching "
                "iteration/step structure; run 0 and run "
                f"{idx} differ."
            )

    merged: List[IterationObservation] = []
    for i in range(len(reference)):
        scores: List[float] = []
        for run in runs:
            scores.extend(run[i].scores)
        meta = dict(reference[i].metadata)
        meta["seeds"] = len(runs)
        merged.append(
            IterationObservation(
                iteration=reference[i].iteration,
                scores=scores,
                higher_is_better=higher_is_better,
                metadata=meta,
            )
        )
    return merged


def observations_from_results_file(
    path: str, higher_is_better: bool = True
) -> Dict[str, List[IterationObservation]]:
    """Load a results JSON file and convert it to observations."""
    with open(path, "r", encoding="utf-8") as fh:
        results = json.load(fh)
    return observations_from_results(results, higher_is_better=higher_is_better)
