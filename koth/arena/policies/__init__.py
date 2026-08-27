"""
The strategies, all on the harness's generic `Bayesian` filter: `n_arm` the
closed-form baselines for any arm count, `koth` the package itself at k = 2
and 3. Everything is re-exported here.
"""

from .koth import Koth, Koth2, Koth3
from .n_arm import Elimination, ExploreThenCommit, Gittins, ProbabilityMatching

ALL: tuple[type, ...] = (
    ExploreThenCommit,
    ProbabilityMatching,
    Elimination,
    Gittins,
    Koth2,
    Koth3,
)
"""The sweep's roster, in report order."""

__all__ = [
    "ALL",
    "Elimination",
    "ExploreThenCommit",
    "Gittins",
    "Koth",
    "Koth2",
    "Koth3",
    "ProbabilityMatching",
]
