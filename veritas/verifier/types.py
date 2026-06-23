"""Data structures exchanged with the VeriTAS Verifier.

The Verifier observes a stream of agent iterations. Each iteration carries one
or more *noisy* objective measurements of a candidate method's performance,
together with optional natural-language artifacts (the method description and
its code) and optional subjective rubric ratings produced by an
LLM-as-a-Judge. The Verifier returns a structured result describing how much
genuine, reproducible progress that iteration contributed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class IterationObservation:
    """A single iteration produced by the research agent.

    Attributes:
        iteration: Zero-based index of this iteration within a run.
        scores: One or more noisy objective measurements of the candidate
            method (for example, the MLRC-Bench metric evaluated under one or
            more seeds). ``higher_is_better`` controls orientation. At least one
            value is required.
        higher_is_better: If ``True`` (default), larger ``scores`` are better.
            If ``False``, scores are internally negated so the rest of the
            pipeline can always assume "higher is better".
        method_text: Optional natural-language description of the proposed
            method, used only by the fusion prior.
        code: Optional code implementation of the method, used only by the
            fusion prior.
        rubric_ratings: Optional mapping from rubric dimension name (for
            example ``"Validity"``) to an integer Likert rating on the 1..5
            scale produced by ``MLAgentBench/LLM_as_a_Judge.py``. When present,
            these ratings are converted into a calibrated prior over the
            improvement contributed by this iteration.
        metadata: Free-form provenance (task name, model, seed, ...). Not used
            by the math; carried through for reporting and debugging.
    """

    iteration: int
    scores: List[float]
    higher_is_better: bool = True
    method_text: Optional[str] = None
    code: Optional[str] = None
    rubric_ratings: Optional[Dict[str, float]] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scores is None or len(self.scores) == 0:
            raise ValueError("IterationObservation requires at least one score.")
        self.scores = [float(s) for s in self.scores]

    def oriented_scores(self) -> List[float]:
        """Return scores oriented so that larger is always better."""
        if self.higher_is_better:
            return list(self.scores)
        return [-s for s in self.scores]


@dataclass
class PosteriorSummary:
    """Compact summary of a Student-t marginal posterior over a quantity.

    The Normal-Inverse-Gamma model yields Student-t marginals. We summarise
    each marginal by its location, scale, and degrees of freedom so downstream
    consumers can recompute tail probabilities without holding the full state.
    """

    mean: float
    scale: float  # scale parameter of the Student-t (not the variance)
    dof: float    # degrees of freedom

    def as_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "scale": self.scale, "dof": self.dof}


@dataclass
class VerifierResult:
    """Verifier output for one iteration.

    Attributes:
        iteration: Index of the scored iteration.
        reward: The additional VeriTAS score for this iteration,
            ``reward = poi * info_gain`` (confidence the gain is real times the
            magnitude of validated information added). Always non-negative.
        poi: Posterior probability that the iteration genuinely improved on the
            incumbent best by more than ``eps_min`` (probability of
            improvement).
        info_gain: Validated information gain in nats: the KL divergence
            between the posterior over the best achievable performance after and
            before incorporating this iteration.
        expected_improvement: Posterior expected positive improvement over the
            incumbent best (Bayesian-optimization acquisition value).
        accept: Decision flag, ``True`` iff ``poi`` exceeds the acceptance
            threshold ``tau``.
        delta_posterior: Summary of the posterior over the improvement
            ``Delta_t`` for this iteration.
        incumbent_posterior: Summary of the posterior over the incumbent best
            performance after this iteration.
        used_rubric_prior: Whether rubric ratings contributed a fusion prior.
        metadata: Provenance carried over from the observation.
    """

    iteration: int
    reward: float
    poi: float
    info_gain: float
    expected_improvement: float
    accept: bool
    delta_posterior: PosteriorSummary
    incumbent_posterior: PosteriorSummary
    used_rubric_prior: bool = False
    metadata: Dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "iteration": self.iteration,
            "reward": self.reward,
            "poi": self.poi,
            "info_gain": self.info_gain,
            "expected_improvement": self.expected_improvement,
            "accept": self.accept,
            "delta_posterior": self.delta_posterior.as_dict(),
            "incumbent_posterior": self.incumbent_posterior.as_dict(),
            "used_rubric_prior": self.used_rubric_prior,
            "metadata": dict(self.metadata),
        }
