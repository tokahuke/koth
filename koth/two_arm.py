"""
The two-arm net, dimensionless: the exploration premium `u(muhat, tauhat)`, the
value `v = relu(muhat) + u`, and the Hamiltonian that reads the policy off it.
Trained on `muhat >= 0`; the premium is even, so a backend evaluates at
`|muhat|` and reflects the policy, and every method here takes either sign.

This module is the contract and the weights. Each backend implements it in its
own realm: `numpy_.two_arm` with central differences, `torch_.two_arm` with
autograd.
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
"""
The flattest `tauhat` the net was trained at; below it the floor's shape is
continued self-similarly (response at the z-preserving floor state, envelope
at the true one).
"""


@cache
def weights() -> dict[str, np.ndarray]:
    """
    The net's parameters as float64 arrays: `log_scale`, `feature_scale`,
    and per layer `w<i>`, `b<i>`, with a `g<i>` gain on every hidden layer.
    """
    return _weights.load("two_arm")


def fixture() -> dict[str, np.ndarray]:
    """
    Reference values pinn computed on 2000 states across the trained decades
    (`tools/from_pinn.py`): `muhat`, `tauhat`, the premium `u`, the learning
    number `l_ab` and the policy `alpha`. What every backend's self-check
    compares against.
    """
    return _weights.load("two_arm.check")


class TwoArm(Net[A]):
    """
    The net on one backend. Shapes broadcast: a leading batch axis on
    `muhat` and `tauhat` comes back on every output.
    """

    K = 2

    @abstractmethod
    def premium(self, muhat: A, tauhat: A) -> A:
        """`u = exp(log_scale) nu(-muhat, tauhat**-1/2) y / (1 + y)`, `y = relu(r)**2`."""

    @abstractmethod
    def value(self, muhat: A, tauhat: A) -> A:
        """`v = relu(muhat) + u`, the dimensionless value."""

    @abstractmethod
    def learning(self, muhat: A, tauhat: A) -> tuple[A, A, A]:
        """
        `(z, s, L_ab)` at `|muhat|`: the chart coordinates `z = muhat sqrt(tauhat)`,
        `s = log tauhat`, and the learning number
        `g_s + g_zz / 2 + z g_z / 2 - g / 2` of `g = e^(s/2) u(z e^(-s/2), e^s)`.
        """

    @abstractmethod
    def policy(self, muhat: A, tauhat: A) -> A:
        """
        The argmax treatment share of the Hamiltonian's quadratic
        `-L x**2 + (e^s z + L) x` on [0, 1]: the stationary point clamped in,
        against both endpoints, read at `|muhat|` and reflected (`1 - alpha`)
        where `muhat < 0`.
        """
