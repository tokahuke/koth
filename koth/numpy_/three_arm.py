"""The three-arm net on numpy, derivatives by central differences."""

from __future__ import annotations

import math
import numpy as np

from scipy import special
from typing import override

from .. import three_arm

STEP = 1e-4
"""
Central-difference step, float64, relative: on the means in units of
`max(1, |m|)`, on the precision entries in units of `det / tau` (the scale on
which the Schur complements move; a step relative to the entry itself moves
them by 10-20% deep in the funnel). Against autograd the learning numbers
agree to 3e-6 at q90 and 3e-4 at q99 (measured 2026-08-27); the tail is the
dead region, the free-boundary band (C^1 premium, as in the two-arm backend)
and correlations above 0.9997, and the policy read off them agrees to 1.5e-5
at q99.
"""

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)
_NODES, _WEIGHTS = 0.5 * (_NODES + 1.0), 0.5 * _WEIGHTS
"""Genz quadrature for the bivariate normal cdf: Gauss-Legendre on [0, 1]."""

_RHO_MAX = 1.0 - 1e-6
"""
How close to +-1 a correlation may come before `asin` and `1 / sqrt(1 - rho**2)`
blow up.
"""


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    """The standard normal density."""
    return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _bivariate_ndtr(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """P(X <= h, Y <= k) for standard normals at correlation `rho`, Genz quadrature."""
    h, k, rho = np.broadcast_arrays(h, k, rho)
    top = np.arcsin(np.clip(rho, -_RHO_MAX, _RHO_MAX))
    theta = top[..., None] * _NODES
    quadratic = (
        h[..., None] ** 2
        + k[..., None] ** 2
        - 2.0 * h[..., None] * k[..., None] * np.sin(theta)
    )
    integral = (np.exp(-quadratic / (2.0 * np.cos(theta) ** 2)) * _WEIGHTS).sum(-1)

    return special.ndtr(h) * special.ndtr(k) + top * integral / (2.0 * math.pi)


def nu2(
    mean_b: np.ndarray,
    mean_c: np.ndarray,
    stddev_b: np.ndarray,
    stddev_c: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    """E[max(0, X, Y)] for bivariate normal (X, Y): the free-information envelope."""
    rho = np.clip(rho, -_RHO_MAX, _RHO_MAX)
    a = np.sqrt((stddev_b - stddev_c) ** 2 + 2.0 * stddev_b * stddev_c * (1.0 - rho))
    root = np.sqrt((1.0 - rho) * (1.0 + rho))
    d = (mean_b - mean_c) / a
    h_b, h_c = mean_b / stddev_b, mean_c / stddev_c
    tilt = (
        stddev_c * mean_b * (stddev_c - rho * stddev_b)
        + stddev_b * mean_c * (stddev_b - rho * stddev_c)
    ) / (stddev_b * stddev_c * a * root)
    expectation = (
        mean_b * _bivariate_ndtr(h_b, d, (stddev_b - rho * stddev_c) / a)
        + mean_c * _bivariate_ndtr(h_c, -d, (stddev_c - rho * stddev_b) / a)
        + stddev_b * _normal_pdf(h_b) * special.ndtr((rho * h_b - h_c) / root)
        + stddev_c * _normal_pdf(h_c) * special.ndtr((rho * h_c - h_b) / root)
        + a * _normal_pdf(d) * special.ndtr(tilt)
    )

    return np.maximum(expectation, 0.0)


def _project(
    tau_bb: np.ndarray, tau_bc: np.ndarray, tau_cc: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The precision projected onto the trained support, in pair coordinates: the
    two largest at least `PRIOR_FLOOR`, the smallest no deeper than the funnel's
    det ceiling. States already inside pass through untouched.
    """
    pairs = np.stack([tau_bb + tau_bc, tau_cc + tau_bc, -tau_bc], -1)
    order = np.argsort(pairs, axis=-1)
    ordered = np.take_along_axis(pairs, order, -1)
    ordered[..., 1:] = np.maximum(ordered[..., 1:], three_arm.PRIOR_FLOOR)
    top, middle = ordered[..., 2], ordered[..., 1]
    det_floor = np.maximum(three_arm.PRIOR_FLOOR**2, three_arm.DET_KEEP * top * middle)
    deepest = (top * middle - det_floor) / (top + middle)
    ordered[..., 0] = np.maximum(ordered[..., 0], -deepest)
    clamped = np.take_along_axis(ordered, np.argsort(order, axis=-1), -1)
    moved = (clamped != pairs).any(-1)

    return (
        np.where(moved, clamped[..., 0] + clamped[..., 2], tau_bb),
        np.where(moved, -clamped[..., 2], tau_bc),
        np.where(moved, clamped[..., 1] + clamped[..., 2], tau_cc),
    )


def _fold(
    m_b: np.ndarray,
    m_c: np.ndarray,
    tau_bb: np.ndarray,
    tau_bc: np.ndarray,
    tau_cc: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """
    Roll a state into the fundamental wedge by the relabel sorting the arm
    levels `(0, m_b, m_c)` descending; the returned `order` maps wedge role k to
    physical arm `order[..., k]`.
    """
    levels = np.stack([np.zeros_like(m_b), m_b, m_c], -1)
    order = np.argsort(-levels, axis=-1, kind="stable")
    sorted_levels = np.take_along_axis(levels, order, -1)
    pairs = np.stack([tau_bb + tau_bc, tau_cc + tau_bc, -tau_bc], -1)

    def pair(arm_one: np.ndarray, arm_two: np.ndarray) -> np.ndarray:
        """The pair coordinate of two arms, at index i + j - 1."""
        return np.take_along_axis(pairs, (arm_one + arm_two - 1)[..., None], -1)[..., 0]

    c_ab = pair(order[..., 0], order[..., 1])
    c_ac = pair(order[..., 0], order[..., 2])
    c_bc = pair(order[..., 1], order[..., 2])
    folded = (
        sorted_levels[..., 1] - sorted_levels[..., 0],
        sorted_levels[..., 2] - sorted_levels[..., 0],
        c_ab + c_bc,
        -c_bc,
        c_ac + c_bc,
    )

    return folded, order


def _maximize(
    c_xx: np.ndarray,
    c_yy: np.ndarray,
    c_xy: np.ndarray,
    c_x: np.ndarray,
    c_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The argmax of `c_xx x**2 + c_yy y**2 + c_xy x y + c_x x + c_y y` over the
    triangle `{x, y >= 0, x + y <= 1}`: interior and edge stationary points and
    the corners, seven candidates, the biggest wins.
    """

    def finite(denominator: np.ndarray) -> np.ndarray:
        """Exact zeros replaced, so dead candidates stay finite and lose."""
        return np.where(np.abs(denominator) < 1e-12, 1.0, denominator)

    zeros, ones = np.zeros_like(c_x), np.ones_like(c_x)
    det = 4.0 * c_xx * c_yy - c_xy**2
    x_interior = (c_xy * c_y - 2.0 * c_yy * c_x) / finite(det)
    y_interior = (c_xy * c_x - 2.0 * c_xx * c_y) / finite(det)
    feasible = (
        (x_interior >= 0)
        & (y_interior >= 0)
        & (x_interior + y_interior <= 1)
        & (np.abs(det) >= 1e-12)
    )
    x_interior, y_interior = np.clip(x_interior, 0.0, 1.0), np.clip(
        y_interior, 0.0, 1.0
    )
    x_leg = np.clip(-c_x / (2.0 * finite(c_xx)), 0.0, 1.0)
    y_leg = np.clip(-c_y / (2.0 * finite(c_yy)), 0.0, 1.0)
    x_hyp = np.clip(
        (2.0 * c_yy - c_xy - c_x + c_y) / (2.0 * finite(c_xx + c_yy - c_xy)), 0.0, 1.0
    )
    x = np.stack([x_interior, x_leg, zeros, x_hyp, zeros, ones, zeros], -1)
    y = np.stack([y_interior, zeros, y_leg, 1.0 - x_hyp, zeros, zeros, ones], -1)
    values = (
        c_xx[..., None] * x**2
        + c_yy[..., None] * y**2
        + c_xy[..., None] * x * y
        + c_x[..., None] * x
        + c_y[..., None] * y
    )
    values[..., 0] = np.where(feasible, values[..., 0], -np.inf)
    best = values.argmax(-1)[..., None]

    return (
        np.take_along_axis(x, best, -1)[..., 0],
        np.take_along_axis(y, best, -1)[..., 0],
    )


class ThreeArm(three_arm.ThreeArm[np.ndarray]):
    """The three-arm net on numpy."""

    def __init__(self) -> None:
        self.w = three_arm.weights()

    @override
    def _premium(
        self,
        m_b: np.ndarray,
        m_c: np.ndarray,
        tau_bb: np.ndarray,
        tau_bc: np.ndarray,
        tau_cc: np.ndarray,
    ) -> np.ndarray:
        """The net as trained, wedge states only."""
        m_b, m_c, tau_bb, tau_bc, tau_cc = np.broadcast_arrays(
            m_b, m_c, tau_bb, tau_bc, tau_cc
        )
        det = tau_bb * tau_cc - tau_bc**2
        precision_b = tau_bb - tau_bc**2 / tau_cc
        precision_c = tau_cc - tau_bc**2 / tau_bb
        precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)
        m_bc = m_b - m_c
        correlation = -tau_bc / np.sqrt(tau_bb * tau_cc)
        scaled = (
            np.stack(
                [
                    m_b,
                    m_c,
                    tau_bb,
                    tau_bc,
                    tau_cc,
                    np.log(precision_b),
                    np.log(precision_c),
                    np.log(precision_bc),
                    m_b * np.sqrt(precision_b),
                    m_c * np.sqrt(precision_c),
                    m_bc * np.sqrt(precision_bc),
                    m_b * precision_b,
                    m_c * precision_c,
                    m_bc * precision_bc,
                    correlation,
                ],
                -1,
            )
            / self.w["feature_scale"]
        )
        x = scaled
        layer = 0

        while f"g{layer}" in self.w:
            x = np.tanh(
                self.w[f"g{layer}"] * (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])
            )
            layer += 1
        response = (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])[..., 0]
        bumps = (
            np.maximum(scaled @ self.w["kink_in_w"].T + self.w["kink_in_b"], 0.0) ** 2
        )
        response = (
            response
            + ((bumps / (1.0 + bumps)) @ self.w["kink_out_w"].T + self.w["kink_out_b"])[
                ..., 0
            ]
        )
        envelope = np.exp(self.w["log_scale"]) * nu2(
            m_b,
            m_c,
            1.0 / np.sqrt(precision_b),
            1.0 / np.sqrt(precision_c),
            correlation,
        )
        y = np.maximum(response, 0.0) ** 2

        return envelope * y / (1.0 + y)

    @override
    def premium(
        self,
        m_b: np.ndarray,
        m_c: np.ndarray,
        tau_bb: np.ndarray,
        tau_bc: np.ndarray,
        tau_cc: np.ndarray,
    ) -> np.ndarray:
        folded, _ = _fold(m_b, m_c, *_project(tau_bb, tau_bc, tau_cc))

        return self._premium(*folded)

    @override
    def value(
        self,
        m_b: np.ndarray,
        m_c: np.ndarray,
        tau_bb: np.ndarray,
        tau_bc: np.ndarray,
        tau_cc: np.ndarray,
    ) -> np.ndarray:
        commit = np.maximum(np.maximum(m_b, m_c), 0.0)

        return commit + self.premium(m_b, m_c, tau_bb, tau_bc, tau_cc)

    @override
    def learning(
        self,
        m_b: np.ndarray,
        m_c: np.ndarray,
        tau_bb: np.ndarray,
        tau_bc: np.ndarray,
        tau_cc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Differences on the premium alone: the commit term is flat inside the
        # wedge, and at the wall autograd reads relu'(0) = 0 where a difference
        # across it would read 1 / h.
        def u(
            d_b: float, d_c: float, d_bb: float, d_bc: float, d_cc: float
        ) -> np.ndarray:
            """The premium at the state displaced by the five offsets."""
            return self._premium(
                m_b + d_b, m_c + d_c, tau_bb + d_bb, tau_bc + d_bc, tau_cc + d_cc
            )

        det = tau_bb * tau_cc - tau_bc**2
        h_b, h_c = STEP * np.maximum(1.0, np.abs(m_b)), STEP * np.maximum(
            1.0, np.abs(m_c)
        )
        h_bb, h_cc = STEP * det / tau_cc, STEP * det / tau_bb
        h_bc = STEP * det / np.sqrt(tau_bb * tau_cc)
        center = u(0.0, 0.0, 0.0, 0.0, 0.0)
        b_up, b_down = u(h_b, 0.0, 0.0, 0.0, 0.0), u(-h_b, 0.0, 0.0, 0.0, 0.0)
        c_up, c_down = u(0.0, h_c, 0.0, 0.0, 0.0), u(0.0, -h_c, 0.0, 0.0, 0.0)
        u_mbmb = (b_up - 2.0 * center + b_down) / h_b**2
        u_mcmc = (c_up - 2.0 * center + c_down) / h_c**2
        u_mbmc = (
            u(h_b, h_c, 0.0, 0.0, 0.0)
            - u(h_b, -h_c, 0.0, 0.0, 0.0)
            - u(-h_b, h_c, 0.0, 0.0, 0.0)
            + u(-h_b, -h_c, 0.0, 0.0, 0.0)
        ) / (4.0 * h_b * h_c)
        u_tbb = (u(0.0, 0.0, h_bb, 0.0, 0.0) - u(0.0, 0.0, -h_bb, 0.0, 0.0)) / (
            2.0 * h_bb
        )
        u_tbc = (u(0.0, 0.0, 0.0, h_bc, 0.0) - u(0.0, 0.0, 0.0, -h_bc, 0.0)) / (
            2.0 * h_bc
        )
        u_tcc = (u(0.0, 0.0, 0.0, 0.0, h_cc) - u(0.0, 0.0, 0.0, 0.0, -h_cc)) / (
            2.0 * h_cc
        )

        def mean_diffusion(d_b: np.ndarray, d_c: np.ndarray) -> np.ndarray:
            """The Ito term `(1/2) d' D2m(u) d` along one pair direction."""
            return 0.5 * (d_b**2 * u_mbmb + 2.0 * d_b * d_c * u_mbmc + d_c**2 * u_mcmc)

        l_ab = mean_diffusion(tau_cc / det, -tau_bc / det) + u_tbb
        l_ac = mean_diffusion(-tau_bc / det, tau_bb / det) + u_tcc
        l_bc = mean_diffusion((tau_cc + tau_bc) / det, -(tau_bb + tau_bc) / det) + (
            u_tbb + u_tcc - u_tbc
        )

        return l_ab, l_ac, l_bc

    @override
    def policy(
        self,
        m_b: np.ndarray,
        m_c: np.ndarray,
        tau_bb: np.ndarray,
        tau_bc: np.ndarray,
        tau_cc: np.ndarray,
    ) -> np.ndarray:
        folded, order = _fold(m_b, m_c, *_project(tau_bb, tau_bc, tau_cc))
        l_ab, l_ac, l_bc = self.learning(*folded)
        x, y = _maximize(
            -l_ab, -l_ac, l_bc - l_ab - l_ac, folded[0] + l_ab, folded[1] + l_ac
        )
        roles = np.stack([1.0 - x - y, x, y], -1)
        allocation = np.zeros_like(roles)
        np.put_along_axis(allocation, order, roles, -1)

        return allocation

    @override
    def subset_value(self, contrast: np.ndarray, precision: np.ndarray) -> np.ndarray:
        return self.value(*_unpack(contrast, precision))

    @override
    def subset_policy(self, contrast: np.ndarray, precision: np.ndarray) -> np.ndarray:
        return self.policy(*_unpack(contrast, precision))


def _unpack(contrast: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, ...]:
    """The subset form as the five chart-form scalars."""
    return (
        contrast[..., 0],
        contrast[..., 1],
        precision[..., 0, 0],
        precision[..., 0, 1],
        precision[..., 1, 1],
    )


if __name__ == "__main__":
    check = three_arm.fixture()
    state = tuple(check[name] for name in ("m_b", "m_c", "tau_bb", "tau_bc", "tau_cc"))
    net = ThreeArm()

    # nu2 at the triple point: ratio to two one-arm premia is (2 + sqrt 2) / 4.
    zero, one = np.zeros(1), np.ones(1)
    triple = nu2(zero, zero, one, one, zero) / (2.0 * one / math.sqrt(2.0 * math.pi))
    assert abs(float(triple[0]) - (2.0 + math.sqrt(2.0)) / 4.0) < 1e-6, triple

    # The premium and value on raw states, against pinn's float32 evaluation.
    u = net.premium(*state)
    gap = np.abs(u - check["u"]) / np.maximum(np.abs(check["u"]), 1e-6)
    assert np.median(gap) < 1e-4 and np.quantile(gap, 0.99) < 1e-2, (
        np.median(gap),
        np.quantile(gap, 0.99),
    )
    value = net.value(*state)
    assert np.allclose(value, check["value"], rtol=1e-3, atol=1e-5), np.abs(
        value - check["value"]
    ).max()

    # Learning numbers on the folded state, and the policy in physical labels.
    folded, _ = _fold(state[0], state[1], *_project(*state[2:]))
    for name, mine in zip(("l_ab", "l_ac", "l_bc"), net.learning(*folded)):
        gap = np.abs(mine - check[name]) / np.maximum(np.abs(check[name]), 1e-3)
        assert np.median(gap) < 1e-3 and np.quantile(gap, 0.95) < 1e-1, (
            name,
            np.median(gap),
            np.quantile(gap, 0.95),
        )
    allocation = net.policy(*state)
    gap = np.abs(allocation - check["allocation"]).max(-1)
    assert np.allclose(allocation.sum(-1), 1.0) and (allocation >= 0.0).all()
    assert np.quantile(gap, 0.99) < 1e-2 and gap.max() < 0.1, (
        np.quantile(gap, 0.99),
        gap.max(),
    )

    # S3 invariance: swapping b and c relabels the policy and keeps the value.
    swapped = (state[1], state[0], state[4], state[3], state[2])
    assert np.allclose(net.value(*swapped), value)
    assert np.allclose(net.policy(*swapped)[..., [0, 2, 1]], allocation, atol=1e-6)
    print(
        f"three_arm on numpy: matches pinn on {len(u)} states "
        f"(policy q99 {np.quantile(gap, 0.99):.1e}, max {gap.max():.1e})"
    )
