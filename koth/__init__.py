"""
King of the hill: the top-k allocation heuristic for A/B/n tests with
correlated arms.

    test = koth.Test(rho=0.001, sigma=450.0)
    decision = test.decide(koth.State(mean, cov), k=3)
    decision.allocation, decision.committed, decision.contenders, decision.value

`Test` holds one experiment's rates, and nothing else: the readout dictionary
between your units and the nets' chart is three scalar factors, so this layer
is written once and takes the backend (`numpy_.decide.Decider` by default,
`torch_.decide.Decider` with the `torch` extra) as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from .decide import Decider, Decision, check_k
from .numpy_.state import State
from .state import State as AnyState

A = TypeVar("A")
"""The backend's array type."""


@dataclass(frozen=True)
class Test(Generic[A]):
    """
    One experiment: a discount rate and a noise level on a shared time unit
    (the epoch), the same `sigma` for every arm.
    """

    rho: float
    """
    Discount rate per epoch, your time preference: `1 / rho` is the number of
    epochs after which a result is worth 37% of one today. Not a deadline.
    """

    sigma: float
    """Noise scale of one arm's estimate over one epoch at full allocation."""

    backend: Decider[A] | None = None
    """The nets to run; None is numpy."""

    def __post_init__(self) -> None:
        if self.rho <= 0.0 or self.sigma <= 0.0:
            raise ValueError(
                f"rho and sigma must be positive, got {self.rho}, {self.sigma}"
            )

    @property
    def decider(self) -> Decider[A]:
        """The backend in use, the numpy one when none was given."""
        if self.backend is not None:
            return self.backend
        from .numpy_.decide import Decider as NumpyDecider

        return NumpyDecider()

    def decide(self, state: AnyState[A], k: int | None = None) -> Decision[A]:
        """
        What to do this epoch: `state` in your units, `k` the contenders
        (2 or 3, `min(3, arms)` by default). Allocations are fractions and the
        value comes back in the units of `mean`.
        """
        k = check_k(k, state.arms)
        mean_scale = self.sigma * self.rho**0.5
        chart = replace(
            state,
            mean=state.mean / mean_scale,
            cov=state.cov / (self.rho * self.sigma**2),
        )
        decision = self.decider.decide(chart, k)

        return replace(decision, value=decision.value * mean_scale / self.rho)

    def observe(self, state: AnyState[A], outcomes: A, allocations: A) -> AnyState[A]:
        """
        The belief after a run of epochs: `outcomes[t, i]` is arm i's metric in
        epoch t as measured, `allocations[t, i]` the share of the allocation it had
        then (0 means no reading, and the value there is ignored). Both
        `(epochs, arms)`; leading batch axes as on `state`.
        """
        weight = allocations.sum(-2)
        estimate = (allocations * outcomes).sum(-2) / (weight + (weight == 0))

        return state.update(estimate, weight / self.sigma**2)


__all__ = ["Decision", "State", "Test"]
