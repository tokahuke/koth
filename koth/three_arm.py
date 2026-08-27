"""
The three-arm net, dimensionless: the exploration premium on states
`(m_b, m_c, tau_bb, tau_bc, tau_cc)`, two contrasts against the control and
their precision matrix, the value `v = relu(max(m_b, m_c)) + u`, and the
Hamiltonian that reads the policy off it. Trained on the fundamental wedge
`{m_c <= m_b <= 0}` only; the premium is S3-invariant, so a backend projects a
state onto the trained support, folds it into the wedge, and un-permutes the
policy back to physical labels.

This module is the contract and the weights. Each backend implements it in its
own realm: `numpy_.three_arm` with central differences, `torch_.three_arm`
with autograd.
"""

from __future__ import annotations

import numpy as np

from abc import abstractmethod
from functools import cache
from typing import TypeVar

from . import _weights
from .net import Net

A = TypeVar("A")
"""The backend's array type."""

PRIOR_FLOOR = 1e-3
"""The floor on the two largest pair precisions of the trained support."""

DET_KEEP = 1e-3
"""
The funnel's relative det floor, `det >= DET_KEEP * (pair_a pair_b)`: how far
negative the smallest pair precision may go on the trained support.
"""


@cache
def weights() -> dict[str, np.ndarray]:
    """
    The net's parameters as float64 arrays: `log_scale`, `feature_scale`,
    per layer `w<i>`, `b<i>` with a `g<i>` gain on every hidden layer, and the
    kink branch `kink_in_w`, `kink_in_b`, `kink_out_w`, `kink_out_b`.
    """
    return _weights.load("three_arm")


def fixture() -> dict[str, np.ndarray]:
    """
    Reference values pinn computed on 2000 states (`tools/from_pinn.py`),
    precisions from its sampler, funnel included, means of either sign: the
    state, its `value` and physical-label `allocation`, and on the folded state
    the premium `u`, the learning numbers `l_ab`, `l_ac`, `l_bc` and the wedge
    policy `x`, `y`.
    """
    return _weights.load("three_arm.check")


class ThreeArm(Net[A]):
    """
    The net on one backend. Shapes broadcast: a leading batch axis on the five
    state arrays comes back on every output.
    """

    K = 3

    @abstractmethod
    def premium(self, m_b: A, m_c: A, tau_bb: A, tau_bc: A, tau_cc: A) -> A:
        """The premium on any positive-definite state, projected and folded."""

    @abstractmethod
    def value(self, m_b: A, m_c: A, tau_bb: A, tau_bc: A, tau_cc: A) -> A:
        """`v = relu(max(m_b, m_c)) + u`, the dimensionless value."""

    @abstractmethod
    def learning(
        self, m_b: A, m_c: A, tau_bb: A, tau_bc: A, tau_cc: A
    ) -> tuple[A, A, A]:
        """
        The pairwise learning numbers `(L_ab, L_ac, L_bc)` on a *wedge* state,
        as given: mean diffusion along the pair's direction plus the precision
        gained in its coordinate.
        """

    @abstractmethod
    def policy(self, m_b: A, m_c: A, tau_bb: A, tau_bc: A, tau_cc: A) -> A:
        """
        The argmax allocation over physical arms, rows `(alpha_a, alpha_b,
        alpha_c)`: the Hamiltonian's quadratic maximized over the triangle on
        the folded state, un-permuted back.
        """
