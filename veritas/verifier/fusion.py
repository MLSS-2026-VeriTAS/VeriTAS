"""Calibrated fusion of LLM-as-a-Judge rubric ratings with objective evidence.

MLRC-Bench ships an ``LLM_as_a_Judge`` that scores a method on a 1..5 Likert
scale across dimensions such as Validity and Rigorousness. Those ratings are
cheap and informative but notoriously over-optimistic and uncalibrated. Rather
than discard them, the Verifier turns them into a *prior* over the improvement a
candidate is expected to deliver, then lets the objective (noisy) measurements
act as the likelihood. The objective evidence can confirm or override the
subjective prior, and the prior regularizes single-shot noisy metrics.

The map from rubric rating to expected gain is linear and *calibratable*: given
historical pairs of (mean rubric rating, observed gain), ``calibrate_rubric_map``
fits the slope/intercept by least squares so the prior is grounded in data
rather than a guess. ``reliability_table`` supports calibration diagnostics
(reliability diagrams).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import NIGPrior, VerifierConfig

NEUTRAL_RATING = 3.0  # midpoint of the 1..5 Likert scale: "no expected change"
RATING_HALF_RANGE = 2.0  # distance from neutral to either extreme


@dataclass
class RubricMap:
    """Calibrated affine map from normalized rubric rating to expected gain.

    ``expected_gain = slope * normalized_rating + intercept`` where
    ``normalized_rating = (mean_rating - 3) / 2`` lies in ``[-1, 1]``.
    """

    slope: float
    intercept: float = 0.0

    def expected_gain(self, mean_rating: float) -> float:
        normalized = (mean_rating - NEUTRAL_RATING) / RATING_HALF_RANGE
        return self.slope * normalized + self.intercept


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def summarize_ratings(ratings: Dict[str, float]) -> Tuple[float, float]:
    """Return the mean rating and a disagreement measure across dimensions.

    Disagreement is the population standard deviation of the per-dimension
    ratings; higher disagreement lowers our confidence in the rubric prior.
    """
    values = [float(v) for v in ratings.values()]
    if len(values) == 0:
        raise ValueError("ratings must contain at least one dimension.")
    mean = _mean(values)
    var = _mean([(v - mean) ** 2 for v in values])
    return mean, var ** 0.5


def rubric_to_prior(
    ratings: Dict[str, float],
    base_mean: float,
    config: VerifierConfig,
    rubric_map: Optional[RubricMap] = None,
) -> NIGPrior:
    """Convert rubric ratings into an NIG prior over the candidate's mean.

    The prior mean is the incumbent ``base_mean`` shifted by the rubric's
    expected gain. The prior strength (pseudo-count ``lam``) grows with rubric
    confidence: confident, internally-consistent ratings pull the candidate
    posterior more strongly, while contradictory ratings contribute a weak
    prior that the objective measurements easily dominate.
    """
    if rubric_map is None:
        rubric_map = RubricMap(slope=config.rubric_gain_scale)

    mean_rating, disagreement = summarize_ratings(ratings)
    expected_gain = rubric_map.expected_gain(mean_rating)

    # Confidence in [0, 1]: maximal when all dimensions agree (disagreement 0),
    # decaying as dimensions spread out across the 1..5 scale.
    confidence = 1.0 / (1.0 + disagreement)
    lam = max(config.rubric_prior_strength * confidence, 1e-3)

    return NIGPrior(
        mu=base_mean + expected_gain,
        lam=lam,
        alpha=config.prior.alpha,
        beta=config.prior.beta,
    )


def calibrate_rubric_map(
    mean_ratings: Sequence[float], observed_gains: Sequence[float]
) -> RubricMap:
    """Least-squares fit of the rating->gain map from historical data.

    Args:
        mean_ratings: Mean rubric rating (1..5) for each past iteration.
        observed_gains: The corresponding objectively-measured gain.

    Returns:
        A ``RubricMap`` whose slope/intercept minimize squared error against the
        normalized ratings. Falls back to a zero-slope (uninformative) map when
        the ratings carry no variation.
    """
    if len(mean_ratings) != len(observed_gains):
        raise ValueError("mean_ratings and observed_gains must be equal length.")
    if len(mean_ratings) < 2:
        raise ValueError("Need at least two points to calibrate.")

    xs = [(r - NEUTRAL_RATING) / RATING_HALF_RANGE for r in mean_ratings]
    ys = list(observed_gains)
    xbar = _mean(xs)
    ybar = _mean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return RubricMap(slope=0.0, intercept=ybar)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    return RubricMap(slope=slope, intercept=intercept)


def reliability_table(
    predicted_probs: Sequence[float],
    outcomes: Sequence[bool],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Bin predicted probabilities against observed frequencies.

    Produces the rows of a reliability diagram used to assess whether the
    Verifier's probability-of-improvement is calibrated (predicted ~ observed).
    """
    if len(predicted_probs) != len(outcomes):
        raise ValueError("predicted_probs and outcomes must be equal length.")

    rows: List[Dict[str, float]] = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [
            i
            for i, p in enumerate(predicted_probs)
            if (p >= lo and (p < hi or (b == n_bins - 1 and p <= hi)))
        ]
        if not idx:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "mean_pred": float("nan"), "observed_freq": float("nan")})
            continue
        mean_pred = _mean([predicted_probs[i] for i in idx])
        observed = _mean([1.0 if outcomes[i] else 0.0 for i in idx])
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "count": float(len(idx)),
                "mean_pred": mean_pred,
                "observed_freq": observed,
            }
        )
    return rows
