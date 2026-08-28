"""
The strategies, all on the harness's generic `Bayesian` filter: `n_arm` the
closed-form baselines for any arm count, `koth` the package itself at k = 2
and 3. Everything is re-exported here.
"""

from .koth import JointThompson, KotH, KotH2, KotH3
from .n_arm import Elimination, ExploreThenCommit, Gittins, ProbabilityMatching

ALL: tuple[type, ...] = (
    ExploreThenCommit,
    ProbabilityMatching,
    Elimination,
    Gittins,
    KotH2,
    KotH3,
    JointThompson,
)
"""The sweep's roster, in report order."""

__all__ = [
    "ALL",
    "Elimination",
    "ExploreThenCommit",
    "Gittins",
    "JointThompson",
    "KotH",
    "KotH2",
    "KotH3",
    "ProbabilityMatching",
]
