"""
What the subset selection asks of a trained net, whatever its arm count: a
value and a policy on the *subset form* of a state, the contrasts of the
subset's arms against its first member and their precision matrix. The
problem contracts (`two_arm`, `three_arm`) declare `K` and add their own chart
form; the backends implement both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

A = TypeVar("A")
"""The backend's array type."""


class Net(ABC, Generic[A]):
    """A trained net on `K` arms, on one backend."""

    K: ClassVar[int]
    """Arms the net was trained on."""

    @abstractmethod
    def subset_value(self, contrast: A, precision: A) -> A:
        """
        The dimensionless value, `(...)`, of a subset with contrast means
        `(..., K - 1)` and contrast precision `(..., K - 1, K - 1)`, relative to
        the subset's first member.
        """

    @abstractmethod
    def subset_policy(self, contrast: A, precision: A) -> A:
        """The argmax allocation over the subset's arms, `(..., K)`, same inputs."""
