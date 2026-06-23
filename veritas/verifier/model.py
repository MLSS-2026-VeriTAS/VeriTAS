"""Closed-form Bayesian machinery for the VeriTAS Verifier.

The Verifier treats each agent iteration as a noisy experiment and reasons
about the latent (noise-free) performance of methods with a conjugate
Normal-Inverse-Gamma (NIG) model. This module provides the small set of
closed-form primitives used by ``verifier.py``:

* ``nig_update``        - conjugate posterior update from noisy scores.
* ``student_t_marginal`` - the Student-t marginal posterior over the mean.
* ``gaussian_moments``  - mean/variance of that Student-t (for tractable
                           differences and KL).
* ``probability_of_improvement`` / ``expected_improvement`` - acquisition
                           quantities over the improvement ``Delta_t``.
* ``kl_gaussian``       - KL between two Gaussians.
* ``improvement_information`` - the one-sided validated information gain:
                           evidence that ``Delta_t`` exceeds zero (positive-only
                           KL-style score against the null).

Everything here is pure (no I/O, no global state) and unit-tested against
hand-computed values.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from scipy.stats import norm

from .config import NIGPrior
from .types import PosteriorSummary

SQRT_2PI = math.sqrt(2.0 * math.pi)


def nig_update(prior: NIGPrior, scores: Sequence[float]) -> NIGPrior:
    """Return the NIG posterior after observing ``scores``.

    Uses the standard conjugate update for a Gaussian likelihood with unknown
    mean and variance. ``scores`` must contain at least one value.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("nig_update requires at least one observation.")

    ybar = sum(scores) / n
    ss = sum((y - ybar) ** 2 for y in scores)  # sum of squared deviations

    lam_n = prior.lam + n
    mu_n = (prior.lam * prior.mu + n * ybar) / lam_n
    alpha_n = prior.alpha + n / 2.0
    beta_n = (
        prior.beta
        + 0.5 * ss
        + 0.5 * (prior.lam * n / lam_n) * (ybar - prior.mu) ** 2
    )
    return NIGPrior(mu=mu_n, lam=lam_n, alpha=alpha_n, beta=beta_n)


def student_t_marginal(posterior: NIGPrior) -> PosteriorSummary:
    """Student-t marginal posterior over the mean ``theta``.

    Under the NIG posterior, ``theta ~ t_{2 alpha}(mu, beta / (lam * alpha))``.
    """
    dof = 2.0 * posterior.alpha
    scale = math.sqrt(posterior.beta / (posterior.lam * posterior.alpha))
    return PosteriorSummary(mean=posterior.mu, scale=scale, dof=dof)


def expected_noise_std(posterior: NIGPrior) -> float:
    """Posterior mean of the measurement-noise standard deviation.

    Under the Inverse-Gamma posterior on ``sigma^2``, ``E[sigma^2] = beta /
    (alpha - 1)`` for ``alpha > 1``. A candidate whose seeds disagree (for
    example, one lucky outlier) yields a large inferred noise, which the
    Verifier uses to raise the bar for crediting an improvement.
    """
    if posterior.alpha > 1.0:
        return math.sqrt(posterior.beta / (posterior.alpha - 1.0))
    return math.sqrt(posterior.beta)


def gaussian_moments(summary: PosteriorSummary, min_variance: float = 1e-9) -> Tuple[float, float]:
    """Mean and variance of a Student-t marginal.

    For ``dof > 2`` the variance is ``scale^2 * dof / (dof - 2)``. ``alpha >= 1``
    in our priors guarantees ``dof > 2``; the fallback only guards pathological
    inputs.
    """
    mean = summary.mean
    if summary.dof > 2.0:
        var = summary.scale ** 2 * summary.dof / (summary.dof - 2.0)
    else:
        var = summary.scale ** 2 * 50.0  # heavy-tailed guard; keep finite
    return mean, max(var, min_variance)


def probability_of_improvement(
    delta_mean: float, delta_var: float, eps_min: float, min_variance: float = 1e-9
) -> float:
    """Posterior probability that the improvement exceeds ``eps_min``.

    ``Delta_t`` is approximated as Gaussian; returns ``P(Delta_t > eps_min)``.
    """
    sigma = math.sqrt(max(delta_var, min_variance))
    z = (delta_mean - eps_min) / sigma
    return float(norm.cdf(z))


def expected_improvement(
    delta_mean: float, delta_var: float, min_variance: float = 1e-9
) -> float:
    """Posterior expected positive improvement ``E[max(Delta_t, 0)]``.

    This is the classic Bayesian-optimization Expected Improvement acquisition
    with the incumbent as the reference, computed in closed form for a Gaussian
    ``Delta_t``.
    """
    sigma = math.sqrt(max(delta_var, min_variance))
    z = delta_mean / sigma
    return float(delta_mean * norm.cdf(z) + sigma * norm.pdf(z))


def kl_gaussian(
    m1: float, v1: float, m0: float, v0: float, min_variance: float = 1e-9
) -> float:
    """KL divergence ``KL(N(m1, v1) || N(m0, v0))`` in nats (non-negative)."""
    v1 = max(v1, min_variance)
    v0 = max(v0, min_variance)
    kl = 0.5 * (math.log(v0 / v1) + (v1 + (m1 - m0) ** 2) / v0 - 1.0)
    return max(kl, 0.0)


def improvement_information(
    delta_mean: float, delta_var: float, min_variance: float = 1e-9
) -> float:
    """Validated information gain for positive improvement evidence (nats).

    Defined as the KL divergence between the improvement posterior
    ``N(delta_mean, delta_var)`` and the same posterior shifted to the
    no-improvement null ``N(0, delta_var)``, but clipped to be one-sided:
    regressions carry zero validated gain. For Gaussians this is
    ``max(delta_mean, 0)^2 / (2 * delta_var)``.
    """
    positive_delta = max(delta_mean, 0.0)
    return kl_gaussian(positive_delta, delta_var, 0.0, delta_var, min_variance)
