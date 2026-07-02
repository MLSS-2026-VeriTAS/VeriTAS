"""VeriTAS: Verifiable and Testable Agents for Science.

This package contains the VeriTAS Verifier, a Bayesian module that scores each
iteration of an iterative ML-research agent (such as MLRC-Bench) by the amount
of *real, reproducible* progress it contributes, separating genuine improvement
from measurement noise.
"""

__all__ = ["verifier"]
