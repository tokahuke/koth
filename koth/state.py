"""
A joint Gaussian belief over the arm effects, and the ways to build one. This
module is the contract; each backend implements it in its own realm
(`numpy_.state`, `torch_.state`), and `koth.State` is the numpy one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

A = TypeVar("A")
"""The backend's array type."""


@dataclass(frozen=True)
class State(ABC, Generic[A]):
    """A joint Gaussian belief over the arm effects, batched over leading axes."""

    mean: A
    """Posterior means, `(..., arms)`."""

    cov: A
    """Posterior covariance, `(..., arms, arms)`, positive definite."""

    def __post_init__(self) -> None:
        arms = self.mean.shape[-1]

        if arms < 2:
            raise ValueError(f"need at least 2 arms, got {arms}")

        if tuple(self.cov.shape) != (*self.mean.shape, arms):
            raise ValueError(
                f"cov must be {(*self.mean.shape, arms)} for mean {self.mean.shape}, "
                f"got {tuple(self.cov.shape)}"
            )

    @property
    def arms(self) -> int:
        """Arm count, control included."""
        return self.mean.shape[-1]

    @classmethod
    @abstractmethod
    def flat(cls, arms: int, std: float) -> Self:
        """Ignorance: every arm at mean 0 and sd `std`, independent."""

    @classmethod
    @abstractmethod
    def independent(cls, mean: A, var: A) -> Self:
        """Per-arm estimates with their variances, `(..., arms)` each, uncorrelated."""

    @abstractmethod
    def update(self, estimate: A, precision: A) -> Self:
        """
        The belief after an estimate per arm, `(..., arms)`, at a precision per
        arm, `(..., arms)`, 0 for an arm without one. `Test.observe` builds these
        from a series of outcomes; this is the step underneath.
        """
