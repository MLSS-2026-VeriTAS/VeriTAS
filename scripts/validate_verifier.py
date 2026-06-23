"""Synthetic paired-ablation validation for the VeriTAS Verifier.

This harness validates the Verifier *without* needing GPUs, API keys, or a full
MLRC-Bench run. It simulates the trajectory of an iterative research agent whose
candidate methods have a *known* ground-truth latent quality, then corrupts the
observed scores with measurement noise and occasional "noise spikes" (a lucky
single seed). Because the ground truth is known, we can measure what no real run
can: how often each acceptance policy is fooled, and how close it gets to the
best achievable method.

Two acceptance policies are compared under identical draws (a paired design,
mirroring the protocol in the project README):

* greedy   - adopt a candidate whenever its mean observed score beats the best
             observed so far (the standard "keep the best-looking" baseline).
* verifier - adopt a candidate only when the Bayesian Verifier judges the
             improvement genuine (probability of improvement above ``tau``).

Reported per seed and in aggregate:
  - final true performance of each policy and the paired net improvement,
  - false-acceptance rate (adopting a candidate that is not truly better),
  - regret against the best achievable method,
  - calibration of the Verifier's probability-of-improvement.

Usage:
    python scripts/validate_verifier.py --seeds 0 1 2 3 4 --compare-verifier
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make the repository root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veritas.verifier import (  # noqa: E402
    IterationObservation,
    NIGPrior,
    Verifier,
    VerifierConfig,
)
from veritas.verifier.fusion import reliability_table  # noqa: E402


@dataclass
class SimConfig:
    """Parameters of the synthetic research-agent trajectory."""

    iterations: int = 60
    n_seeds: int = 4
    sigma: float = 0.03          # per-seed measurement noise (std)
    q0: float = 0.50             # starting true performance
    p_improve: float = 0.40      # probability a candidate is a genuine gain
    improve_scale: float = 0.05  # mean size of a genuine gain
    lateral_scale: float = 0.02  # size of benign no-op moves
    p_deceptive: float = 0.20    # probability of a deceptive regression
    reg_min: float = 0.05        # a deceptive change is genuinely this much worse
    reg_max: float = 0.15
    spike_min: float = 0.18      # lucky-seed outlier that makes it *look* better
    spike_max: float = 0.40
    use_rubric: bool = False     # also feed correlated rubric ratings


@dataclass
class Draw:
    """Policy-independent random draw for one iteration (common random numbers).

    The agent proposes a modification to its *currently deployed* method. The
    modification has a real, unobserved effect, observed only through noisy
    seed measurements (occasionally corrupted by a lucky-seed spike). Because
    the draw does not depend on the policy, both policies are compared on
    identical randomness while still being free to diverge through their own
    accept/reject decisions and the resulting compounding.
    """

    effect: float            # true additive change vs the deployed method
    noise: List[float]       # per-seed measurement noise
    spike_index: int         # which seed receives a spike (if any)
    spike_value: float       # spike magnitude (0.0 if no spike)
    rubric_bias: float       # latent rubric optimism for this iteration


@dataclass
class SeedResult:
    seed: int
    greedy_final_true: float
    verifier_final_true: float
    oracle_true: float
    net_improvement: float
    greedy_false_accept_rate: float
    verifier_false_accept_rate: float
    greedy_regret: float
    verifier_regret: float
    greedy_mean_regret: float = 0.0
    verifier_mean_regret: float = 0.0
    poi_values: List[float] = field(default_factory=list)
    poi_labels: List[bool] = field(default_factory=list)
    greedy_trajectory: List[float] = field(default_factory=list)
    verifier_trajectory: List[float] = field(default_factory=list)
    oracle_trajectory: List[float] = field(default_factory=list)


def simulate_draws(rng: np.random.Generator, cfg: SimConfig) -> List[Draw]:
    """Generate policy-independent draws (common random numbers).

    Three iteration types:
      * genuine    - a real positive effect, consistent across seeds.
      * deceptive  - a real *negative* effect (a regression) that a single lucky
                     seed makes look like a large improvement. The case the
                     Verifier is designed to catch.
      * benign     - a small no-op/regression with no deceptive spike.
    """
    draws: List[Draw] = []
    for _ in range(cfg.iterations):
        r = rng.random()
        spike_value = 0.0
        if r < cfg.p_improve:
            effect = float(rng.exponential(cfg.improve_scale))
        elif r < cfg.p_improve + cfg.p_deceptive:
            effect = -float(rng.uniform(cfg.reg_min, cfg.reg_max))
            spike_value = float(rng.uniform(cfg.spike_min, cfg.spike_max))
        else:
            effect = -float(abs(rng.normal(0.0, cfg.lateral_scale)))

        noise = list(rng.normal(0.0, cfg.sigma, size=cfg.n_seeds))
        spike_index = int(rng.integers(0, cfg.n_seeds))
        rubric_bias = float(rng.normal(0.0, 0.6))
        draws.append(Draw(effect, noise, spike_index, spike_value, rubric_bias))
    return draws


def _scores_for(deployed_true: float, draw: Draw) -> Tuple[float, List[float]]:
    """Realize a candidate's true quality and noisy observations for a draw."""
    cand_true = deployed_true + draw.effect
    scores = [cand_true + n for n in draw.noise]
    if draw.spike_value > 0.0:
        scores[draw.spike_index] += draw.spike_value
    return cand_true, scores


def oracle_curve(cfg: SimConfig, draws: List[Draw]) -> List[float]:
    """Deployed true performance of an omniscient policy (accept iff effect>0)."""
    deployed = cfg.q0
    curve: List[float] = []
    for d in draws:
        if d.effect > 0.0:
            deployed += d.effect
        curve.append(deployed)
    return curve


def run_greedy(cfg: SimConfig, draws: List[Draw]) -> Tuple[List[float], int, int]:
    """Greedy 'accept if it looks better than the current method' policy.

    Builds on its own deployed method, so accepting a noise spike permanently
    corrupts the base for all future iterations (compounding).
    """
    deployed_true = cfg.q0
    deployed_obs = cfg.q0
    trajectory: List[float] = []
    accepts = 0
    false_accepts = 0
    for d in draws:
        cand_true, scores = _scores_for(deployed_true, d)
        cand_mean = float(np.mean(scores))
        if cand_mean > deployed_obs:
            if d.effect <= 0.0:
                false_accepts += 1
            deployed_true = cand_true
            deployed_obs = cand_mean
            accepts += 1
        trajectory.append(deployed_true)
    return trajectory, accepts, false_accepts


def run_verifier(
    cfg: SimConfig, draws: List[Draw], config: VerifierConfig, use_rubric: bool
) -> Tuple[List[float], int, int, List[float], List[bool]]:
    """Verifier-gated policy under the same draws. Returns trajectory, accepts,
    false accepts, and paired (poi, genuine-improvement) data for calibration."""
    verifier = Verifier(config=config)
    deployed_true = cfg.q0
    trajectory: List[float] = []
    accepts = 0
    false_accepts = 0
    poi_values: List[float] = []
    poi_labels: List[bool] = []

    for i, d in enumerate(draws):
        cand_true, scores = _scores_for(deployed_true, d)
        rubric = None
        if use_rubric:
            base = 3.5 + (1.5 if d.effect > 0.0 else -0.5)
            rubric = {
                dim: float(np.clip(round(base + d.rubric_bias), 1, 5))
                for dim in ("Validity", "Rigorousness", "Innovativeness")
            }
        result = verifier.update(IterationObservation(iteration=i, scores=scores, rubric_ratings=rubric))

        if i > 0:
            poi_values.append(result.poi)
            poi_labels.append(d.effect > 0.0)

        if result.accept:
            if d.effect <= 0.0:
                false_accepts += 1
            deployed_true = cand_true
            accepts += 1
        trajectory.append(deployed_true)

    return trajectory, accepts, false_accepts, poi_values, poi_labels


def make_config(cfg: SimConfig) -> VerifierConfig:
    """A Verifier config matched to the simulated noise level."""
    alpha = 2.0
    # Set prior so the prior expected noise variance is ~ sigma^2. Use a weak
    # prior on the mean (small lam) so observed scores dominate and genuine
    # gains are not shrunk toward the incumbent.
    beta = cfg.sigma ** 2 * (alpha - 1.0)
    return VerifierConfig(
        prior=NIGPrior(mu=cfg.q0, lam=0.3, alpha=alpha, beta=beta),
        eps_min=0.0,
        eps_min_rel=0.4,
        tau=0.65,
    )


def run_seed(seed: int, cfg: SimConfig) -> SeedResult:
    rng = np.random.default_rng(seed)
    draws = simulate_draws(rng, cfg)
    oracle = oracle_curve(cfg, draws)
    oracle_final = oracle[-1]

    g_traj, g_acc, g_false = run_greedy(cfg, draws)
    config = make_config(cfg)
    v_traj, v_acc, v_false, poi_vals, poi_labels = run_verifier(cfg, draws, config, cfg.use_rubric)

    g_final = g_traj[-1]
    v_final = v_traj[-1]

    # False-accept rates are normalized by the number of accepts each made.
    g_far = g_false / max(1, g_acc)
    v_far = v_false / max(1, v_acc)

    # Mean regret integrates the gap to the oracle over the whole run, so a
    # policy that spends time on spike-corrupted methods is penalized even if
    # it later recovers.
    g_mean_regret = float(np.mean([o - t for o, t in zip(oracle, g_traj)]))
    v_mean_regret = float(np.mean([o - t for o, t in zip(oracle, v_traj)]))

    return SeedResult(
        seed=seed,
        greedy_final_true=g_final,
        verifier_final_true=v_final,
        oracle_true=oracle_final,
        net_improvement=v_final - g_final,
        greedy_false_accept_rate=g_far,
        verifier_false_accept_rate=v_far,
        greedy_regret=oracle_final - g_final,
        verifier_regret=oracle_final - v_final,
        greedy_mean_regret=g_mean_regret,
        verifier_mean_regret=v_mean_regret,
        poi_values=poi_vals,
        poi_labels=poi_labels,
        greedy_trajectory=g_traj,
        verifier_trajectory=v_traj,
        oracle_trajectory=oracle,
    )


def aggregate_and_report(results: List[SeedResult]) -> None:
    print("\n" + "=" * 72)
    print("VeriTAS Verifier - paired ablation (synthetic ground truth)")
    print("=" * 72)
    header = f"{'seed':>5} {'greedy':>10} {'verifier':>10} {'oracle':>10} {'net':>10} {'g_FAR':>8} {'v_FAR':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.seed:>5} {r.greedy_final_true:>10.4f} {r.verifier_final_true:>10.4f} "
            f"{r.oracle_true:>10.4f} {r.net_improvement:>+10.4f} "
            f"{r.greedy_false_accept_rate:>8.2f} {r.verifier_false_accept_rate:>8.2f}"
        )

    nets = [r.net_improvement for r in results]
    g_reg = [r.greedy_regret for r in results]
    v_reg = [r.verifier_regret for r in results]
    g_mreg = [r.greedy_mean_regret for r in results]
    v_mreg = [r.verifier_mean_regret for r in results]
    g_far = [r.greedy_false_accept_rate for r in results]
    v_far = [r.verifier_false_accept_rate for r in results]

    def ms(xs: List[float]) -> str:
        return f"{mean(xs):+.4f} +/- {pstdev(xs):.4f}" if len(xs) > 1 else f"{mean(xs):+.4f}"

    print("-" * len(header))
    print(f"Aggregate over {len(results)} seed(s):")
    print(f"  Net improvement (verifier - greedy final true): {ms(nets)}")
    print(f"  Greedy   final regret vs oracle:                {ms(g_reg)}")
    print(f"  Verifier final regret vs oracle:                {ms(v_reg)}")
    print(f"  Greedy   mean regret over run:                  {ms(g_mreg)}")
    print(f"  Verifier mean regret over run:                  {ms(v_mreg)}")
    print(f"  Greedy   false-acceptance rate:                 {ms(g_far)}")
    print(f"  Verifier false-acceptance rate:                 {ms(v_far)}")

    wins = sum(1 for n in nets if n > 0)
    reg_wins = sum(1 for r in results if r.verifier_mean_regret <= r.greedy_mean_regret)
    print(f"  Seeds where verifier final >= greedy:           {sum(1 for n in nets if n >= 0)}/{len(nets)}")
    print(f"  Seeds where verifier strictly wins (final):     {wins}/{len(nets)}")
    print(f"  Seeds where verifier mean-regret <= greedy:     {reg_wins}/{len(results)}")
    print("=" * 72)


def save_plots(results: List[SeedResult], out_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []

    # 1) Example trajectory (first seed).
    r0 = results[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(r0.oracle_trajectory, color="gray", linestyle="--", label="oracle")
    ax.plot(r0.greedy_trajectory, label="greedy", marker="o", markersize=3)
    ax.plot(r0.verifier_trajectory, label="verifier", marker="s", markersize=3)
    ax.set_xlabel("iteration")
    ax.set_ylabel("deployed true performance")
    ax.set_title(f"Deployed true performance (seed {r0.seed})")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(out_dir, "trajectory.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    saved.append(p1)

    # 2) Reliability diagram for the Verifier's probability of improvement.
    all_poi = [p for r in results for p in r.poi_values]
    all_lab = [l for r in results for l in r.poi_labels]
    if all_poi:
        rows = reliability_table(all_poi, all_lab, n_bins=10)
        xs = [row["mean_pred"] for row in rows if row["count"] > 0]
        ys = [row["observed_freq"] for row in rows if row["count"] > 0]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        ax.plot(xs, ys, marker="o", label="verifier POI")
        ax.set_xlabel("predicted P(improvement)")
        ax.set_ylabel("observed frequency")
        ax.set_title("Reliability diagram")
        ax.legend()
        fig.tight_layout()
        p2 = os.path.join(out_dir, "reliability.png")
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        saved.append(p2)

    # 3) Net improvement across seeds.
    fig, ax = plt.subplots(figsize=(7, 4))
    seeds = [str(r.seed) for r in results]
    nets = [r.net_improvement for r in results]
    ax.bar(seeds, nets)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("seed")
    ax.set_ylabel("net improvement (verifier - greedy)")
    ax.set_title("Paired net improvement by seed")
    fig.tight_layout()
    p3 = os.path.join(out_dir, "net_improvement.png")
    fig.savefig(p3, dpi=120)
    plt.close(fig)
    saved.append(p3)

    return saved


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Interface compatible with the README (accepted; synthetic mode ignores
    # the real-run-specific arguments).
    parser.add_argument("--task", type=str, default="synthetic", help="(compat) task name; synthetic mode ignores this")
    parser.add_argument("--model", type=str, default="synthetic", help="(compat) model name; synthetic mode ignores this")
    parser.add_argument("--gpu-id", type=str, default="0", help="(compat) ignored in synthetic mode")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="random seeds for paired runs")
    parser.add_argument("--compare-verifier", action="store_true", help="run greedy vs verifier comparison (default behavior)")
    parser.add_argument("--iterations", type=int, default=SimConfig.iterations)
    parser.add_argument("--n-seeds", type=int, default=SimConfig.n_seeds, help="measurements (seeds) per candidate")
    parser.add_argument("--use-rubric", action="store_true", help="feed correlated rubric ratings to the verifier")
    parser.add_argument("--out-dir", type=str, default="verifier_validation_out")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = SimConfig(
        iterations=args.iterations,
        n_seeds=args.n_seeds,
        use_rubric=args.use_rubric,
    )

    results = [run_seed(seed, cfg) for seed in args.seeds]
    aggregate_and_report(results)

    if not args.no_plots:
        saved = save_plots(results, args.out_dir)
        print("\nSaved plots:")
        for p in saved:
            print(f"  {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
