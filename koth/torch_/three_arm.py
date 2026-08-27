"""The three-arm net on torch, derivatives by autograd."""

from __future__ import annotations

import math
import numpy as np

try:
    import torch
except ImportError as e:
    e.add_note("consider using the koth[torch] extra")

    raise

from torch import Tensor
from typing import override

from .. import three_arm

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)
_NODES, _WEIGHTS = 0.5 * (_NODES + 1.0), 0.5 * _WEIGHTS
"""Genz quadrature for the bivariate normal cdf: Gauss-Legendre on [0, 1]."""

_RHO_MAX = 1.0 - 1e-6
"""
How close to +-1 a correlation may come before `asin` and `1 / sqrt(1 - rho**2)`
blow up.
"""


def _normal_pdf(x: Tensor) -> Tensor:
    """The standard normal density."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def _bivariate_ndtr(h: Tensor, k: Tensor, rho: Tensor) -> Tensor:
    """P(X <= h, Y <= k) for standard normals at correlation `rho`, Genz quadrature."""
    h, k, rho = torch.broadcast_tensors(h, k, rho)
    nodes = torch.as_tensor(_NODES, dtype=h.dtype, device=h.device)
    weights = torch.as_tensor(_WEIGHTS, dtype=h.dtype, device=h.device)
    top = torch.asin(rho.clamp(-_RHO_MAX, _RHO_MAX))
    theta = top[..., None] * nodes
    quadratic = (
        h[..., None] ** 2
        + k[..., None] ** 2
        - 2.0 * h[..., None] * k[..., None] * torch.sin(theta)
    )
    integral = (torch.exp(-quadratic / (2.0 * torch.cos(theta) ** 2)) * weights).sum(-1)

    return torch.special.ndtr(h) * torch.special.ndtr(k) + top * integral / (
        2.0 * math.pi
    )


def nu2(
    mean_b: Tensor, mean_c: Tensor, stddev_b: Tensor, stddev_c: Tensor, rho: Tensor
) -> Tensor:
    """E[max(0, X, Y)] for bivariate normal (X, Y): the free-information envelope."""
    rho = rho.clamp(-_RHO_MAX, _RHO_MAX)
    a = torch.sqrt((stddev_b - stddev_c) ** 2 + 2.0 * stddev_b * stddev_c * (1.0 - rho))
    root = torch.sqrt((1.0 - rho) * (1.0 + rho))
    d = (mean_b - mean_c) / a
    h_b, h_c = mean_b / stddev_b, mean_c / stddev_c
    tilt = (
        stddev_c * mean_b * (stddev_c - rho * stddev_b)
        + stddev_b * mean_c * (stddev_b - rho * stddev_c)
    ) / (stddev_b * stddev_c * a * root)
    expectation = (
        mean_b * _bivariate_ndtr(h_b, d, (stddev_b - rho * stddev_c) / a)
        + mean_c * _bivariate_ndtr(h_c, -d, (stddev_c - rho * stddev_b) / a)
        + stddev_b * _normal_pdf(h_b) * torch.special.ndtr((rho * h_b - h_c) / root)
        + stddev_c * _normal_pdf(h_c) * torch.special.ndtr((rho * h_c - h_b) / root)
        + a * _normal_pdf(d) * torch.special.ndtr(tilt)
    )

    return expectation.clamp_min(0.0)


def _project(tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor) -> tuple[Tensor, ...]:
    """
    The precision projected onto the trained support, in pair coordinates: the
    two largest at least `PRIOR_FLOOR`, the smallest no deeper than the funnel's
    det ceiling. States already inside pass through untouched.
    """
    pairs = torch.stack([tau_bb + tau_bc, tau_cc + tau_bc, -tau_bc], -1)
    ordered, order = pairs.sort(dim=-1)
    ordered[..., 1:] = ordered[..., 1:].clamp_min(three_arm.PRIOR_FLOOR)
    top, middle = ordered[..., 2], ordered[..., 1]
    det_floor = torch.maximum(
        torch.full_like(top, three_arm.PRIOR_FLOOR**2),
        three_arm.DET_KEEP * top * middle,
    )
    deepest = (top * middle - det_floor) / (top + middle)
    ordered[..., 0] = ordered[..., 0].clamp_min(-deepest)
    clamped = ordered.gather(-1, order.argsort(dim=-1))
    moved = (clamped != pairs).any(dim=-1)

    return (
        torch.where(moved, clamped[..., 0] + clamped[..., 2], tau_bb),
        torch.where(moved, -clamped[..., 2], tau_bc),
        torch.where(moved, clamped[..., 1] + clamped[..., 2], tau_cc),
    )


def _fold(
    m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
) -> tuple[tuple[Tensor, ...], Tensor]:
    """
    Roll a state into the fundamental wedge by the relabel sorting the arm
    levels `(0, m_b, m_c)` descending; the returned `order` maps wedge role k to
    physical arm `order[..., k]`.
    """
    levels = torch.stack([torch.zeros_like(m_b), m_b, m_c], -1)
    order = levels.argsort(dim=-1, descending=True, stable=True)
    sorted_levels = levels.gather(-1, order)
    pairs = torch.stack([tau_bb + tau_bc, tau_cc + tau_bc, -tau_bc], -1)

    def pair(arm_one: Tensor, arm_two: Tensor) -> Tensor:
        """The pair coordinate of two arms, at index i + j - 1."""
        return pairs.gather(-1, (arm_one + arm_two - 1)[..., None])[..., 0]

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
    c_xx: Tensor, c_yy: Tensor, c_xy: Tensor, c_x: Tensor, c_y: Tensor
) -> tuple[Tensor, Tensor]:
    """
    The argmax of `c_xx x**2 + c_yy y**2 + c_xy x y + c_x x + c_y y` over the
    triangle `{x, y >= 0, x + y <= 1}`: interior and edge stationary points and
    the corners, seven candidates, the biggest wins.
    """

    def finite(denominator: Tensor) -> Tensor:
        """Exact zeros replaced, so dead candidates stay finite and lose."""
        return denominator.masked_fill(denominator.abs() < 1e-12, 1.0)

    zeros, ones = torch.zeros_like(c_x), torch.ones_like(c_x)
    det = 4.0 * c_xx * c_yy - c_xy**2
    x_interior = (c_xy * c_y - 2.0 * c_yy * c_x) / finite(det)
    y_interior = (c_xy * c_x - 2.0 * c_xx * c_y) / finite(det)
    feasible = (
        (x_interior >= 0)
        & (y_interior >= 0)
        & (x_interior + y_interior <= 1)
        & (det.abs() >= 1e-12)
    )
    x_interior, y_interior = x_interior.clamp(0.0, 1.0), y_interior.clamp(0.0, 1.0)
    x_leg = (-c_x / (2.0 * finite(c_xx))).clamp(0.0, 1.0)
    y_leg = (-c_y / (2.0 * finite(c_yy))).clamp(0.0, 1.0)
    x_hyp = (
        (2.0 * c_yy - c_xy - c_x + c_y) / (2.0 * finite(c_xx + c_yy - c_xy))
    ).clamp(0.0, 1.0)
    x = torch.stack([x_interior, x_leg, zeros, x_hyp, zeros, ones, zeros], -1)
    y = torch.stack([y_interior, zeros, y_leg, 1.0 - x_hyp, zeros, zeros, ones], -1)
    values = (
        c_xx[..., None] * x**2
        + c_yy[..., None] * y**2
        + c_xy[..., None] * x * y
        + c_x[..., None] * x
        + c_y[..., None] * y
    )
    values[..., 0] = torch.where(feasible, values[..., 0], -torch.inf)
    best = values.argmax(-1, keepdim=True)

    return x.gather(-1, best)[..., 0], y.gather(-1, best)[..., 0]


class ThreeArm(three_arm.ThreeArm[Tensor]):
    """The three-arm net on torch, at `dtype` (float64 by default)."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        self.w = {
            name: torch.as_tensor(value, dtype=dtype)
            for name, value in three_arm.weights().items()
        }

    @override
    def _premium(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        """The net as trained, wedge states only."""
        m_b, m_c, tau_bb, tau_bc, tau_cc = torch.broadcast_tensors(
            m_b, m_c, tau_bb, tau_bc, tau_cc
        )
        det = tau_bb * tau_cc - tau_bc**2
        precision_b = tau_bb - tau_bc**2 / tau_cc
        precision_c = tau_cc - tau_bc**2 / tau_bb
        precision_bc = det / (tau_bb + tau_cc + 2.0 * tau_bc)
        m_bc = m_b - m_c
        correlation = -tau_bc / (tau_bb * tau_cc).sqrt()
        scaled = (
            torch.stack(
                [
                    m_b,
                    m_c,
                    tau_bb,
                    tau_bc,
                    tau_cc,
                    precision_b.log(),
                    precision_c.log(),
                    precision_bc.log(),
                    m_b * precision_b.sqrt(),
                    m_c * precision_c.sqrt(),
                    m_bc * precision_bc.sqrt(),
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
            x = (
                self.w[f"g{layer}"] * (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])
            ).tanh()
            layer += 1
        response = (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])[..., 0]
        bumps = (scaled @ self.w["kink_in_w"].T + self.w["kink_in_b"]).relu() ** 2
        response = (
            response
            + ((bumps / (1.0 + bumps)) @ self.w["kink_out_w"].T + self.w["kink_out_b"])[
                ..., 0
            ]
        )
        envelope = self.w["log_scale"].exp() * nu2(
            m_b, m_c, precision_b.rsqrt(), precision_c.rsqrt(), correlation
        )
        y = response.relu() ** 2

        return envelope * y / (1.0 + y)

    @override
    def premium(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        folded, _ = _fold(m_b, m_c, *_project(tau_bb, tau_bc, tau_cc))

        return self._premium(*folded)

    @override
    def value(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        return torch.maximum(m_b, m_c).relu() + self.premium(
            m_b, m_c, tau_bb, tau_bc, tau_cc
        )

    @override
    def learning(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = [
            x.detach().requires_grad_(True) for x in (m_b, m_c, tau_bb, tau_bc, tau_cc)
        ]
        m_b, m_c, tau_bb, tau_bc, tau_cc = state
        u = self._premium(*state)
        u_mb, u_mc, u_tbb, u_tbc, u_tcc = torch.autograd.grad(
            u.sum(), state, create_graph=True, allow_unused=True, materialize_grads=True
        )
        u_mbmb, u_mbmc = torch.autograd.grad(
            u_mb.sum(),
            [m_b, m_c],
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        (u_mcmc,) = torch.autograd.grad(
            u_mc.sum(),
            [m_c],
            create_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        det = tau_bb * tau_cc - tau_bc**2

        def mean_diffusion(d_b: Tensor, d_c: Tensor) -> Tensor:
            """The Ito term `(1/2) d' D2m(u) d` along one pair direction."""
            return 0.5 * (d_b**2 * u_mbmb + 2.0 * d_b * d_c * u_mbmc + d_c**2 * u_mcmc)

        l_ab = mean_diffusion(tau_cc / det, -tau_bc / det) + u_tbb
        l_ac = mean_diffusion(-tau_bc / det, tau_bb / det) + u_tcc
        l_bc = mean_diffusion((tau_cc + tau_bc) / det, -(tau_bb + tau_bc) / det) + (
            u_tbb + u_tcc - u_tbc
        )

        return l_ab.detach(), l_ac.detach(), l_bc.detach()

    @override
    def policy(
        self, m_b: Tensor, m_c: Tensor, tau_bb: Tensor, tau_bc: Tensor, tau_cc: Tensor
    ) -> Tensor:
        folded, order = _fold(m_b, m_c, *_project(tau_bb, tau_bc, tau_cc))
        l_ab, l_ac, l_bc = self.learning(*folded)
        x, y = _maximize(
            -l_ab, -l_ac, l_bc - l_ab - l_ac, folded[0] + l_ab, folded[1] + l_ac
        )
        roles = torch.stack([1.0 - x - y, x, y], -1)

        return torch.zeros_like(roles).scatter_(-1, order, roles)

    @override
    def subset_value(self, contrast: Tensor, precision: Tensor) -> Tensor:
        return self.value(*_unpack(contrast, precision))

    @override
    def subset_policy(self, contrast: Tensor, precision: Tensor) -> Tensor:
        return self.policy(*_unpack(contrast, precision))


def _unpack(contrast: Tensor, precision: Tensor) -> tuple[Tensor, ...]:
    """The subset form as the five chart-form scalars."""
    return (
        contrast[..., 0],
        contrast[..., 1],
        precision[..., 0, 0],
        precision[..., 0, 1],
        precision[..., 1, 1],
    )


if __name__ == "__main__":
    from ..numpy_.three_arm import ThreeArm as NumpyThreeArm

    check = three_arm.fixture()
    names = ("m_b", "m_c", "tau_bb", "tau_bc", "tau_cc")
    state = tuple(torch.as_tensor(check[name]) for name in names)
    net = ThreeArm()

    # Against pinn's float32 evaluation on the raw states.
    u = net.premium(*state).numpy()
    gap = np.abs(u - check["u"]) / np.maximum(np.abs(check["u"]), 1e-6)
    assert np.median(gap) < 1e-4 and np.quantile(gap, 0.99) < 1e-2, (
        np.median(gap),
        np.quantile(gap, 0.99),
    )
    value = net.value(*state).numpy()
    assert np.allclose(value, check["value"], rtol=1e-3, atol=1e-5)
    folded, _ = _fold(state[0], state[1], *_project(*state[2:]))
    learning = [x.numpy() for x in net.learning(*folded)]

    for name, mine in zip(("l_ab", "l_ac", "l_bc"), learning):
        gap = np.abs(mine - check[name]) / np.maximum(np.abs(check[name]), 1e-3)
        assert np.median(gap) < 1e-4 and np.quantile(gap, 0.95) < 1e-2, (
            name,
            np.median(gap),
            np.quantile(gap, 0.95),
        )
    allocation = net.policy(*state).numpy()
    gap = np.abs(allocation - check["allocation"]).max(-1)
    assert np.quantile(gap, 0.99) < 1e-2 and gap.max() < 0.1, (
        np.quantile(gap, 0.99),
        gap.max(),
    )

    # Against the numpy backend: differences vs autograd agree to 3e-6 at q90
    # (measured 2026-08-27); the tail is the dead region, the free-boundary
    # band and correlations above 0.9997, where the policies still agree.
    numpy_net = NumpyThreeArm()
    numpy_state = tuple(check[name] for name in names)
    assert np.allclose(numpy_net.premium(*numpy_state), u, rtol=1e-10, atol=1e-12)
    folded_np = tuple(x.numpy() for x in folded)

    for name, mine, theirs in zip(
        ("l_ab", "l_ac", "l_bc"), learning, numpy_net.learning(*folded_np)
    ):
        gap = np.abs(mine - theirs) / np.maximum(np.abs(mine), 1e-3)
        assert np.quantile(gap, 0.9) < 1e-4 and np.quantile(gap, 0.99) < 1e-2, (
            name,
            np.quantile(gap, 0.9),
            np.quantile(gap, 0.99),
        )
    gap = np.abs(allocation - numpy_net.policy(*numpy_state)).max(-1)
    assert np.quantile(gap, 0.99) < 1e-3 and gap.max() < 5e-2, (
        np.quantile(gap, 0.99),
        gap.max(),
    )
    print(
        f"three_arm on torch: matches pinn and numpy on {len(u)} states (policy max {gap.max():.1e})"
    )
