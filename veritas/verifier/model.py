"""Minimal EB ranking helpers for verifier candidates."""

from __future__ import annotations

import math
from collections import defaultdict

from veritas.verifier.types import CandidateEvidence, EffectiveCandidate


TAU2_FLOOR = 1e-12


def cluster_effective_candidates(
    evidences: list[CandidateEvidence],
) -> list[EffectiveCandidate]:
    grouped: dict[str, list[CandidateEvidence]] = defaultdict(list)
    for evidence in evidences:
        grouped[evidence.logical_candidate_id].append(evidence)

    effective: list[EffectiveCandidate] = []
    for cluster_id, members in sorted(grouped.items()):
        if len(members) == 1:
            member = members[0]
            effective.append(
                EffectiveCandidate(
                    cluster_id=cluster_id,
                    member_ids=[member.candidate_id],
                    delta_hat=member.delta_hat,
                    total_se=member.total_se,
                )
            )
            continue

        valid_se = all(member.total_se > 0 for member in members)
        flags = ["near_duplicate_cluster"]
        if valid_se:
            weights = [1.0 / (member.total_se**2) for member in members]
            delta_hat = sum(
                weight * member.delta_hat
                for weight, member in zip(weights, members, strict=True)
            ) / sum(weights)
            base_var = 1.0 / sum(weights)
        else:
            flags.append("low_information_cluster_estimate")
            delta_hat = sum(member.delta_hat for member in members) / len(members)
            base_var = _mean([member.total_se**2 for member in members])

        spread = _sample_variance([member.delta_hat for member in members])
        effective.append(
            EffectiveCandidate(
                cluster_id=cluster_id,
                member_ids=[member.candidate_id for member in members],
                delta_hat=delta_hat,
                total_se=math.sqrt(max(base_var + spread, 0.0)),
                flags=flags,
            )
        )

    return effective


def estimate_eb_parameters(
    candidates: list[EffectiveCandidate],
    *,
    tau2_floor: float = TAU2_FLOOR,
) -> dict:
    if not candidates:
        raise ValueError("EB parameter estimation requires candidates")

    weights = [
        1.0 / max(candidate.total_se**2, tau2_floor)
        for candidate in candidates
    ]
    mu = sum(
        weight * candidate.delta_hat
        for weight, candidate in zip(weights, candidates, strict=True)
    ) / sum(weights)

    delta_variance = _sample_variance([candidate.delta_hat for candidate in candidates])
    mean_se2 = _mean([candidate.total_se**2 for candidate in candidates])
    tau2_raw = delta_variance - mean_se2
    flags: list[str] = []
    if tau2_raw <= 0:
        flags.append("tau2_collapse")
    if len(candidates) < 5:
        flags.append("eb_ranking_only")

    return {
        "mu": mu,
        "tau2_raw": tau2_raw,
        "tau2": max(tau2_raw, tau2_floor),
        "flags": flags,
    }


def rank_candidates(
    candidates: list[EffectiveCandidate],
    *,
    epsilon: float,
    lambda_lower: float = 1.0,
    lambda_upper: float = 1.0,
    z_resolution: float = 2.0,
    uncertainty_high_sd_multiplier: float = 1.0,
) -> dict:
    params = estimate_eb_parameters(candidates)
    tau2 = params["tau2"]
    reports = []

    for candidate in candidates:
        se2 = candidate.total_se**2
        weight = tau2 / (tau2 + se2) if tau2 + se2 > 0 else 0.0
        posterior_mean = weight * candidate.delta_hat + (1.0 - weight) * params["mu"]
        posterior_var = 1.0 / ((1.0 / max(se2, TAU2_FLOOR)) + (1.0 / tau2))
        posterior_sd = math.sqrt(max(posterior_var, 0.0))
        dev_resolution_ratio = epsilon / candidate.total_se if candidate.total_se > 0 else math.inf
        uncertainty_high = posterior_sd > uncertainty_high_sd_multiplier * epsilon

        reports.append(
            {
                "cluster_id": candidate.cluster_id,
                "member_ids": candidate.member_ids,
                "raw_delta": candidate.delta_hat,
                "total_se": candidate.total_se,
                "posterior_mean_delta": posterior_mean,
                "posterior_sd": posterior_sd,
                "lower_evidence_gain": posterior_mean - lambda_lower * posterior_sd,
                "upper_evidence_gain": posterior_mean + lambda_upper * posterior_sd,
                "dev_resolution_ratio": dev_resolution_ratio,
                "dev_resolution_ok": dev_resolution_ratio >= z_resolution,
                "uncertainty_high": uncertainty_high,
                "flags": sorted(set(candidate.flags + params["flags"])),
            }
        )

    reports.sort(key=lambda item: item["posterior_mean_delta"], reverse=True)
    return {
        "mu": params["mu"],
        "tau2_raw": params["tau2_raw"],
        "tau2": params["tau2"],
        "flags": params["flags"],
        "candidates": reports,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)
