"""VeriTAS Verifier: Bayesian Validated Information Gain.

Public API:

    from veritas.verifier import Verifier, VerifierConfig
    from veritas.verifier import IterationObservation, VerifierResult

See ``docs/verifier_method.md`` for the precise probabilistic model.
"""

from .config import NIGPrior, VerifierConfig
from .fusion import RubricMap, calibrate_rubric_map, reliability_table
from .types import IterationObservation, PosteriorSummary, VerifierResult
from .verifier import Verifier

__all__ = [
    "Verifier",
    "VerifierConfig",
    "NIGPrior",
    "RubricMap",
    "calibrate_rubric_map",
    "reliability_table",
    "IterationObservation",
    "VerifierResult",
    "PosteriorSummary",
]
