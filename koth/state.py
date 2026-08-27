"""
A joint Gaussian belief over the arm effects, and the ways to build and update
one. This module is the contract; each backend implements it in its own realm
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
    def observe(self, estimate: A, precision: A) -> Self:
        """
        The belief after one epoch's evidence: `estimate[i]` for arm i at
        `precision[i]` (for `n` visitors with outcome sd `s`, `n / s**2`; 0 means
        not observed). The conjugate update in information form, so a correlated
        prior moves the unobserved arms too.
        """
