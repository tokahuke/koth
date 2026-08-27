"""
The decision on a joint belief over any number of arms, dimensionless: value
every k-subset by the k-arm net (relative to the subset's first member, whose
commit value is added back), play the net's policy on the argmax subset and
nothing on the rest. `k` is 2 or 3, one trained net each.

This module is the contract. Each backend implements it in its own realm
(`numpy_.decide`, `torch_.decide`); `Test` binds one to an experiment's units.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from .state import State

A = TypeVar("A")
"""The backend's array type."""


@dataclass(frozen=True)
class Decision(Generic[A]):
    """What to do this epoch, batched like the state it answers."""

    allocation: A
    """Traffic share per arm, `(..., arms)`, a simplex row; a vertex is a commit."""

    committed: A
    """The arm committed to, `(...)`, or -1 while the test is still running."""

    contenders: A
    """The k arms the allocation is spread over, `(..., k)`, in subset order."""

    value: A
    """The value of continuing from here, `(...)`, dimensionless."""


class Decider(ABC, Generic[A]):
    """The subset selection on one backend."""

    @abstractmethod
    def decide(self, state: State[A], k: int) -> Decision[A]:
        """The decision on a dimensionless state; `k` is 2 or 3 and at most `arms`."""


def check_k(k: int | None, arms: int) -> int:
    """`k` resolved and validated: `min(3, arms)` by default, 2 or 3, at most `arms`."""
    k = min(3, arms) if k is None else k

    if k not in (2, 3):
        raise ValueError(f"k must be 2 or 3, got {k}")

    if k > arms:
        raise ValueError(f"k = {k} needs at least {k} arms, got {arms}")

    return k
