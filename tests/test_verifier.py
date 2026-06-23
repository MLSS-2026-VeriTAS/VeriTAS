"""Unit and integration tests for the VeriTAS Verifier.

These tests run with no GPU/API and cover:
  * closed-form correctness of the Normal-Inverse-Gamma update and the model
    primitives (POI, EI, Clark max, KL),
  * monotonicity of the probability of improvement,
  * information gain ordering (genuine shift > noise-only),
  * rubric fusion shifting the posterior in the expected direction,
  * the MLRC-Bench adapter parsing a synthetic results payload,
  * an end-to-end check that the Verifier rejects an injected noise spike but
    accepts a genuine improvement.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veritas.verifier import (  # noqa: E402
    IterationObservation,
    NIGPrior,
    Verifier,
    VerifierConfig,
    calibrate_rubric_map,
    reliability_table,
)
from veritas.verifier import model  # noqa: E402
from veritas.verifier import fusion  # noqa: E402
from veritas.verifier import mlrc_adapter  # noqa: E402


# --------------------------------------------------------------------------- #
# Closed-form model primitives
# --------------------------------------------------------------------------- #

def test_nig_update_matches_closed_form():
    prior = NIGPrior(mu=0.0, lam=1.0, alpha=2.0, beta=1.0)
    scores = [1.0, 3.0]  # ybar = 2.0, ss = 2.0
    post = model.nig_update(prior, scores)

    assert post.lam == pytest.approx(3.0)
    assert post.mu == pytest.approx((1 * 0.0 + 2 * 2.0) / 3.0)
    assert post.alpha == pytest.approx(2.0 + 1.0)
    expected_beta = 1.0 + 0.5 * 2.0 + 0.5 * (1.0 * 2.0 / 3.0) * (2.0 - 0.0) ** 2
    assert post.beta == pytest.approx(expected_beta)


def test_student_t_marginal_and_moments():
    post = NIGPrior(mu=2.0, lam=3.0, alpha=3.0, beta=6.0)
    summ = model.student_t_marginal(post)
    assert summ.mean == pytest.approx(2.0)
    assert summ.dof == pytest.approx(6.0)
    assert summ.scale == pytest.approx(math.sqrt(6.0 / (3.0 * 3.0)))

    mean, var = model.gaussian_moments(summ)
    assert mean == pytest.approx(2.0)
    assert var == pytest.approx(summ.scale ** 2 * summ.dof / (summ.dof - 2.0))


def test_probability_of_improvement_is_monotonic():
    # Larger observed gain -> higher POI; tighter variance -> higher POI.
    p_small = model.probability_of_improvement(0.1, 1.0, 0.0)
    p_large = model.probability_of_improvement(0.5, 1.0, 0.0)
    assert p_large > p_small

    p_wide = model.probability_of_improvement(0.5, 4.0, 0.0)
    assert p_large > p_wide  # tighter posterior is more confident

    # At delta == eps the probability is exactly 0.5.
    assert model.probability_of_improvement(0.3, 1.0, 0.3) == pytest.approx(0.5)


def test_expected_improvement_non_negative_and_increasing():
    ei_low = model.expected_improvement(-0.5, 1.0)
    ei_high = model.expected_improvement(0.5, 1.0)
    assert ei_low >= 0.0
    assert ei_high >= 0.0
    assert ei_high > ei_low


def test_kl_gaussian_zero_for_identical_and_positive_otherwise():
    assert model.kl_gaussian(1.0, 2.0, 1.0, 2.0) == pytest.approx(0.0, abs=1e-12)
    assert model.kl_gaussian(2.0, 1.0, 0.0, 1.0) > 0.0


def test_improvement_information_is_squared_snr():
    # IG = max(delta_mean, 0)^2 / (2 * delta_var); one-sided in delta_mean.
    assert model.improvement_information(0.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert model.improvement_information(1.0, 0.5) == pytest.approx(1.0, rel=1e-9)
    assert model.improvement_information(-1.0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert model.improvement_information(2.0, 0.5) > model.improvement_information(1.0, 0.5)
    # More uncertainty (larger variance) lowers the information for a fixed gain.
    assert model.improvement_information(1.0, 1.0) < model.improvement_information(1.0, 0.5)


# --------------------------------------------------------------------------- #
# Information gain ordering
# --------------------------------------------------------------------------- #

def test_info_gain_genuine_exceeds_noise():
    cfg = VerifierConfig(prior=NIGPrior(mu=0.5, lam=1.0, alpha=2.0, beta=0.001),
                         eps_min_rel=0.5, tau=0.7)

    # Establish an incumbent around 0.5.
    v_noise = Verifier(config=cfg)
    v_noise.update(IterationObservation(0, [0.50, 0.51, 0.49]))
    noise_res = v_noise.update(IterationObservation(1, [0.50, 0.49, 0.51]))

    v_gain = Verifier(config=cfg)
    v_gain.update(IterationObservation(0, [0.50, 0.51, 0.49]))
    gain_res = v_gain.update(IterationObservation(1, [0.70, 0.71, 0.69]))

    assert gain_res.info_gain > noise_res.info_gain
    assert gain_res.poi > noise_res.poi
    assert gain_res.reward > noise_res.reward


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #

def test_rubric_prior_shifts_mean_in_expected_direction():
    cfg = VerifierConfig()
    high = fusion.rubric_to_prior({"Validity": 5, "Rigorousness": 5}, base_mean=0.5, config=cfg)
    low = fusion.rubric_to_prior({"Validity": 1, "Rigorousness": 1}, base_mean=0.5, config=cfg)
    neutral = fusion.rubric_to_prior({"Validity": 3, "Rigorousness": 3}, base_mean=0.5, config=cfg)

    assert high.mu > neutral.mu
    assert low.mu < neutral.mu
    assert neutral.mu == pytest.approx(0.5)


def test_rubric_disagreement_lowers_prior_strength():
    cfg = VerifierConfig()
    agree = fusion.rubric_to_prior({"A": 5, "B": 5, "C": 5}, base_mean=0.0, config=cfg)
    disagree = fusion.rubric_to_prior({"A": 1, "B": 5, "C": 3}, base_mean=0.0, config=cfg)
    assert agree.lam > disagree.lam


def test_calibrate_rubric_map_recovers_slope():
    # Construct gains that are exactly slope * normalized_rating.
    slope = 0.08
    ratings = [1.0, 2.0, 3.0, 4.0, 5.0]
    gains = [slope * (r - 3.0) / 2.0 for r in ratings]
    rm = calibrate_rubric_map(ratings, gains)
    assert rm.slope == pytest.approx(slope, rel=1e-6)
    assert rm.intercept == pytest.approx(0.0, abs=1e-9)


def test_reliability_table_bins_correctly():
    preds = [0.05, 0.15, 0.95, 0.92]
    outcomes = [False, False, True, True]
    rows = reliability_table(preds, outcomes, n_bins=10)
    assert len(rows) == 10
    populated = [r for r in rows if r["count"] > 0]
    assert sum(r["count"] for r in populated) == 4


# --------------------------------------------------------------------------- #
# MLRC adapter
# --------------------------------------------------------------------------- #

def test_adapter_parses_evaluation_result():
    result = {
        "score": [0.40, 0.55, 0.60],
        "score_steps": [0, 5],  # two intermediate, plus a trailing final
    }
    obs = mlrc_adapter.observations_from_evaluation_result(result)
    assert len(obs) == 3
    assert obs[0].scores == [0.40]
    assert obs[0].metadata["step"] == 0
    assert obs[1].metadata["step"] == 5
    assert obs[-1].metadata["step"] == "final"
    assert obs[-1].scores == [0.60]


def test_adapter_extracts_rubric_ratings():
    judge = {
        "without_code": {
            "Validity": {"Rating": 4, "Review": "x", "Feedback": "y"},
            "Clarity": {"Rating": 5},
        },
        "with_code": {"Validity": {"Rating": 2}},
    }
    ratings = mlrc_adapter.rubric_ratings_from_judge(judge, use_code=False)
    assert ratings == {"Validity": 4.0, "Clarity": 5.0}
    code_ratings = mlrc_adapter.rubric_ratings_from_judge(judge, use_code=True)
    assert code_ratings == {"Validity": 2.0}


def test_merge_seed_observations_gathers_scores():
    runs = [
        [IterationObservation(0, [0.5]), IterationObservation(1, [0.6])],
        [IterationObservation(0, [0.52]), IterationObservation(1, [0.58])],
    ]
    merged = mlrc_adapter.merge_seed_observations(runs)
    assert len(merged) == 2
    assert sorted(merged[0].scores) == [0.5, 0.52]
    assert merged[0].metadata["seeds"] == 2


def test_merge_seed_observations_rejects_mismatched_steps():
    runs = [
        [
            IterationObservation(0, [0.5], metadata={"step": 0}),
            IterationObservation(1, [0.6], metadata={"step": 5}),
        ],
        [
            IterationObservation(0, [0.51], metadata={"step": 0}),
            IterationObservation(1, [0.59], metadata={"step": 6}),
        ],
    ]
    with pytest.raises(ValueError, match="matching iteration/step structure"):
        mlrc_adapter.merge_seed_observations(runs)


# --------------------------------------------------------------------------- #
# End-to-end behavior
# --------------------------------------------------------------------------- #

def test_verifier_rejects_noise_spike_accepts_genuine_gain():
    cfg = VerifierConfig(
        prior=NIGPrior(mu=0.5, lam=1.0, alpha=2.0, beta=0.0009),
        eps_min_rel=0.5,
        tau=0.7,
    )
    v = Verifier(config=cfg)

    # Iteration 0: establish incumbent ~0.50 with low noise.
    v.update(IterationObservation(0, [0.50, 0.51, 0.49]))

    # Iteration 1: a noise spike - one lucky seed is very high, others flat.
    spike = v.update(IterationObservation(1, [0.50, 0.49, 0.95]))
    assert not spike.accept, "verifier should not be fooled by a single lucky seed"

    # Iteration 2: a genuine, reproducible improvement across all seeds.
    gain = v.update(IterationObservation(2, [0.70, 0.71, 0.69]))
    assert gain.accept, "verifier should accept a consistent improvement"
    assert gain.reward > spike.reward


def test_first_iteration_initializes_incumbent():
    v = Verifier()
    assert v.incumbent is None
    res = v.update(IterationObservation(0, [0.5, 0.5, 0.5]))
    assert v.incumbent is not None
    assert v.incumbent.mean == pytest.approx(res.incumbent_posterior.mean)


def test_first_iteration_respects_poi_threshold():
    cfg = VerifierConfig(prior=NIGPrior(mu=0.0, lam=1.0, alpha=2.0, beta=0.001), tau=0.95)
    v = Verifier(config=cfg)
    res = v.update(IterationObservation(0, scores=[-2.0, -2.1, -1.9]))
    assert not res.accept


def test_reward_is_non_negative_over_random_trajectory():
    import random

    rng = random.Random(0)
    v = Verifier()
    for i in range(30):
        scores = [0.5 + rng.gauss(0, 0.05) for _ in range(3)]
        res = v.update(IterationObservation(i, scores))
        assert res.reward >= 0.0
        assert 0.0 <= res.poi <= 1.0
        assert res.info_gain >= 0.0
