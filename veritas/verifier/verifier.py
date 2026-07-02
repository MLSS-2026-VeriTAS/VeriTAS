"""The stateful VeriTAS Verifier.

The Verifier consumes a stream of :class:`IterationObservation` objects and, for
each, returns a :class:`VerifierResult` quantifying how much *validated*
progress the iteration contributed. It maintains a Gaussian belief over the best
achievable latent performance (the incumbent ``theta*``) across iterations.

Per iteration ``t`` the pipeline is:

1. Build the candidate's NIG prior. When rubric ratings are present they are
   fused into a prior centered on the incumbent plus the rubric's expected gain
   (:mod:`veritas.verifier.fusion`); otherwise a weak prior centered on the
   incumbent is used.
2. Conjugate-update that prior with the iteration's noisy scores to obtain the
   candidate posterior over ``theta_t`` (a Student-t, summarized by Gaussian
   moments).
3. Form the improvement ``Delta_t = theta_t - theta*_{t-1}`` (Gaussian) and
   compute the probability of improvement ``POI`` and expected improvement
   ``EI``.
4. Form the "if-accepted" belief about the best achievable performance via the
   moment-matched maximum ``M = max(theta*_{t-1}, theta_t)`` (Clark) and the
   validated information gain ``IG = KL(M || theta*_{t-1})``.
5. Emit reward ``r_t = POI * IG`` (expected validated information) and accept
   iff ``POI > tau``.
6. Persist a *conservative* incumbent update: a ``POI``-gated mixture of ``M``
   and the previous incumbent, so unconfident noise spikes barely move the
   running belief.
"""

from __future__ import annotations

from typing import List, Optional

from . import model
from .config import NIGPrior, VerifierConfig
from .fusion import RubricMap, rubric_to_prior
from .types import (
    IterationObservation,
    PosteriorSummary,
    VerifierResult,
)


class Verifier:
    """Sequential Bayesian verifier of agent research progress."""

    def __init__(
        self,
        config: Optional[VerifierConfig] = None,
        rubric_map: Optional[RubricMap] = None,
    ) -> None:
        self.config = config or VerifierConfig()
        self.rubric_map = rubric_map

        # Belief over the best achievable latent performance (theta*).
        # Initialized from the global prior; re-seeded on the first iteration.
        self._incumbent_mean: Optional[float] = None
        self._incumbent_var: Optional[float] = None
        self._initialized = False
        self.history: List[VerifierResult] = []

    @property
    def incumbent(self) -> Optional[PosteriorSummary]:
        """Current Gaussian belief over ``theta*`` as a summary (dof = inf)."""
        if not self._initialized:
            return None
        scale = (self._incumbent_var or 0.0) ** 0.5
        return PosteriorSummary(mean=self._incumbent_mean or 0.0, scale=scale, dof=float("inf"))

    def _candidate_posterior(self, obs: IterationObservation):
        """Return (mean, var, used_rubric) for the candidate's theta posterior."""
        cfg = self.config
        base_mean = self._incumbent_mean if self._initialized else None

        used_rubric = False
        if obs.rubric_ratings and base_mean is not None:
            prior = rubric_to_prior(obs.rubric_ratings, base_mean, cfg, self.rubric_map)
            used_rubric = True
        else:
            # Weak prior centered on the incumbent (or the global prior mean for
            # the very first iteration), inheriting only the spread parameters.
            mu0 = base_mean if base_mean is not None else cfg.prior.mu
            prior = NIGPrior(mu=mu0, lam=cfg.prior.lam, alpha=cfg.prior.alpha, beta=cfg.prior.beta)

        posterior = model.nig_update(prior, obs.oriented_scores())
        summary = model.student_t_marginal(posterior)
        mean, var = model.gaussian_moments(summary, cfg.min_variance)
        noise_std = model.expected_noise_std(posterior)
        return mean, var, used_rubric, noise_std

    def update(self, obs: IterationObservation) -> VerifierResult:
        """Score a single iteration and update the incumbent belief."""
        cfg = self.config
        cand_mean, cand_var, used_rubric, noise_std = self._candidate_posterior(obs)

        # Minimal effect size scales with the inferred measurement noise: an
        # improvement must beat a fraction of the noise to be credited.
        eps = max(cfg.eps_min, cfg.eps_min_rel * noise_std)

        if not self._initialized:
            # First iteration establishes the incumbent. There is no prior
            # incumbent to improve upon, so credit the information of going from
            # the global prior to this first calibrated belief.
            prior_summary = model.student_t_marginal(cfg.prior)
            prior_mean, prior_var = model.gaussian_moments(prior_summary, cfg.min_variance)

            delta_mean = cand_mean - prior_mean
            delta_var = cand_var + prior_var
            poi = model.probability_of_improvement(delta_mean, delta_var, eps, cfg.min_variance)
            ei = model.expected_improvement(delta_mean, delta_var, cfg.min_variance)

            ig = model.improvement_information(delta_mean, delta_var, cfg.min_variance)
            reward = poi * ig
            accept = poi > cfg.tau

            # Always initialize the incumbent state after the first observation.
            # The deployed frontier only advances when the POI rule accepts.
            if accept:
                self._incumbent_mean = cand_mean
                self._incumbent_var = cand_var
            else:
                self._incumbent_mean = prior_mean
                self._incumbent_var = prior_var
            self._initialized = True

            result = VerifierResult(
                iteration=obs.iteration,
                reward=reward,
                poi=poi,
                info_gain=ig,
                expected_improvement=ei,
                accept=accept,
                delta_posterior=PosteriorSummary(delta_mean, delta_var ** 0.5, float("inf")),
                incumbent_posterior=self.incumbent,  # type: ignore[arg-type]
                used_rubric_prior=used_rubric,
                metadata=dict(obs.metadata),
            )
            self.history.append(result)
            return result

        inc_mean = self._incumbent_mean
        inc_var = self._incumbent_var
        assert inc_mean is not None and inc_var is not None

        # Improvement posterior Delta_t = theta_t - theta*_{t-1}.
        delta_mean = cand_mean - inc_mean
        delta_var = cand_var + inc_var
        poi = model.probability_of_improvement(delta_mean, delta_var, eps, cfg.min_variance)
        ei = model.expected_improvement(delta_mean, delta_var, cfg.min_variance)

        # Validated information gain: evidence the iteration genuinely improved
        # on the incumbent. Reward is that evidence weighted by our confidence.
        ig = model.improvement_information(delta_mean, delta_var, cfg.min_variance)
        reward = poi * ig

        accept = poi > cfg.tau

        # The deployed frontier only advances on an accepted improvement: the
        # accepted candidate's posterior becomes the new incumbent belief.
        # Rejected (for example noise-spike) iterations leave the frontier
        # untouched, so a single lucky run cannot corrupt the incumbent.
        if accept:
            self._incumbent_mean = cand_mean
            self._incumbent_var = cand_var

        result = VerifierResult(
            iteration=obs.iteration,
            reward=reward,
            poi=poi,
            info_gain=ig,
            expected_improvement=ei,
            accept=accept,
            delta_posterior=PosteriorSummary(delta_mean, delta_var ** 0.5, float("inf")),
            incumbent_posterior=self.incumbent,  # type: ignore[arg-type]
            used_rubric_prior=used_rubric,
            metadata=dict(obs.metadata),
        )
        self.history.append(result)
        return result

    def run(self, observations: List[IterationObservation]) -> List[VerifierResult]:
        """Score a full sequence of iterations in order."""
        return [self.update(obs) for obs in observations]
