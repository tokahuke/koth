"""
The policy arena: discrete-epoch simulation of an N-arm environment, discounted
regret against an oracle, and the strategies. Core in harness.py, strategies in
policies/. Needs the `arena` extra (torch).
"""

from .harness import (
    Bayesian,
    Environment,
    Normal,
    Params,
    Policy,
    Run,
    Study,
    UnsupportedNumberOfArms,
    optimal_deadline,
)

__all__ = [
    "Bayesian",
    "Environment",
    "Normal",
    "Params",
    "Policy",
    "Run",
    "Study",
    "UnsupportedNumberOfArms",
    "optimal_deadline",
]
