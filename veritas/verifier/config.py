"""Configuration for the VeriTAS Verifier.

All tunable hyperparameters live here so that experiments (and the paired
ablation in ``scripts/validate_verifier.py``) can vary them in one place.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NIGPrior:
    """Normal-Inverse-Gamma prior over (mean, variance) of a method's scores.

    The prior is ``sigma^2 ~ InvGamma(alpha, beta)`` and
    ``theta | sigma^2 ~ N(mu, sigma^2 / lam)``. ``lam`` acts as a prior sample
    size (pseudo-count) for the mean, while ``alpha``/``beta`` encode the prior
    over the measurement noise. ``alpha`` is kept at or above 1 so the marginal
    Student-t over the mean always has finite variance (dof = 2*alpha > 2).
    """

    mu: float = 0.0
    lam: float = 1.0
    alpha: float = 2.0
    beta: float = 1.0


@dataclass
class VerifierConfig:
    """Hyperparameters controlling the Verifier's decisions and scoring.

    Attributes:
        prior: Default Normal-Inverse-Gamma prior used for a candidate method
            when no rubric prior is supplied. ``mu`` is re-centered on the
            current incumbent at update time, so only the spread parameters of
            this prior matter in practice.
        eps_min: Absolute minimal effect size (in score units) that counts as a
            genuine improvement. Used by the probability-of-improvement test.
        eps_min_rel: Minimal effect size expressed as a multiple of the
            *inferred measurement-noise* standard deviation of the candidate.
            The effective threshold is ``max(eps_min, eps_min_rel *
            noise_std)``, so candidates whose seeds disagree (noise spikes)
            must show a larger mean gain before improvement is credited.
        tau: Acceptance threshold on the probability of improvement. An
            iteration is accepted iff ``poi > tau``.
        rubric_gain_scale: Maps a centered, normalized rubric rating in
            ``[-1, 1]`` to an expected improvement in score units (the slope of
            the calibrated rubric->gain map).
        rubric_prior_strength: Base pseudo-count (``lam``) contributed by a
            confident rubric. Scaled down when rubric dimensions disagree.
        min_variance: Numerical floor on any variance to avoid divide-by-zero.
    """

    prior: NIGPrior = None  # type: ignore[assignment]
    eps_min: float = 0.0
    eps_min_rel: float = 0.0
    tau: float = 0.75
    rubric_gain_scale: float = 0.05
    rubric_prior_strength: float = 2.0
    min_variance: float = 1e-9

    def __post_init__(self) -> None:
        if self.prior is None:
            self.prior = NIGPrior()
