"""The two-arm net on torch, derivatives by autograd."""

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

from .. import two_arm


def nu(mean: Tensor, stddev: Tensor) -> Tensor:
    """E[max(0, X)] for X ~ N(mean, stddev**2), the free-information envelope."""
    z = mean / stddev
    expectation = mean * torch.special.ndtr(z) + stddev * torch.exp(
        -0.5 * z**2
    ) / math.sqrt(2.0 * math.pi)

    return expectation.clamp_min(0.0)


class TwoArm(two_arm.TwoArm[Tensor]):
    """The two-arm net on torch, at `dtype` (float64 by default)."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        self.w = {
            name: torch.as_tensor(value, dtype=dtype)
            for name, value in two_arm.weights().items()
        }

    @override
    def premium(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        return self._premium(muhat.abs(), tauhat)

    @override
    def _premium(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        """The net as trained, `muhat >= 0` only."""
        muhat, tauhat = torch.broadcast_tensors(muhat, tauhat)
        tau_eff = tauhat.clamp_min(two_arm.PRIOR_FLOOR)
        mu_eff = muhat * (tauhat / tau_eff).sqrt()
        x = (
            torch.stack(
                [mu_eff, tau_eff.log(), mu_eff * tau_eff.sqrt(), mu_eff * tau_eff], -1
            )
            / self.w["feature_scale"]
        )
        layer = 0

        while f"g{layer}" in self.w:
            x = (
                self.w[f"g{layer}"] * (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])
            ).tanh()
            layer += 1
        response = (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])[..., 0]
        y = response.relu() ** 2

        return nu(-muhat, tauhat.rsqrt()) * self.w["log_scale"].exp() * y / (1.0 + y)

    @override
    def value(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        return muhat.relu() + self.premium(muhat, tauhat)

    @override
    def _chart(self, z: Tensor, s: Tensor) -> Tensor:
        """The premium on the similarity chart, `g = e^(s/2) u(z e^(-s/2), e^s)`."""
        return (s / 2.0).exp() * self._premium(z * (-s / 2.0).exp(), s.exp())

    @override
    def learning(self, muhat: Tensor, tauhat: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        z = (muhat.abs() * tauhat.sqrt()).detach().requires_grad_(True)
        s = tauhat.log().detach().requires_grad_(True)
        g = self._chart(z, s)
        g_z, g_s = torch.autograd.grad(g.sum(), [z, s], create_graph=True)
        (g_zz,) = torch.autograd.grad(g_z.sum(), z, create_graph=True)

        return z, s, g_s + 0.5 * g_zz + 0.5 * z * g_z - 0.5 * g

    @override
    def policy(self, muhat: Tensor, tauhat: Tensor) -> Tensor:
        z, s, l_ab = self.learning(muhat, tauhat)
        c_xx, c_x = -l_ab, s.exp() * z + l_ab
        safe = c_xx.masked_fill(c_xx.abs() < 1e-12, 1.0)
        vertex = (-c_x / (2.0 * safe)).clamp(0.0, 1.0)
        candidates = torch.stack(
            [vertex, torch.zeros_like(vertex), torch.ones_like(vertex)], -1
        )
        values = c_xx[..., None] * candidates**2 + c_x[..., None] * candidates

        share = candidates.gather(-1, values.argmax(-1, keepdim=True))[..., 0].detach()

        return torch.where(muhat >= 0.0, share, 1.0 - share)

    @override
    def subset_value(self, contrast: Tensor, precision: Tensor) -> Tensor:
        return self.value(contrast[..., 0], precision[..., 0, 0])

    @override
    def subset_policy(self, contrast: Tensor, precision: Tensor) -> Tensor:
        share = self.policy(contrast[..., 0], precision[..., 0, 0])

        return torch.stack([1.0 - share, share], -1)


if __name__ == "__main__":
    from ..numpy_.two_arm import TwoArm as NumpyTwoArm

    check = two_arm.fixture()
    muhat = torch.as_tensor(check["muhat"])
    tauhat = torch.as_tensor(check["tauhat"])
    net = TwoArm()

    # Against pinn's float32 evaluation, and against the numpy backend: the
    # learning number to q99 (the numpy boundary band is O(1e-2) off, by
    # design), the policy everywhere.
    u = net.premium(muhat, tauhat).numpy()
    assert np.allclose(u, check["u"], rtol=1e-4, atol=1e-6), np.abs(
        u - check["u"]
    ).max()
    _, _, l_ab = net.learning(muhat, tauhat)
    l_ab = l_ab.detach().numpy()
    gap = np.abs(l_ab - check["l_ab"]) / np.maximum(np.abs(check["l_ab"]), 1e-3)
    assert np.quantile(gap, 0.99) < 1e-4, np.quantile(gap, 0.99)
    alpha = net.policy(muhat, tauhat).numpy()
    assert np.abs(alpha - check["alpha"]).max() < 1e-3, np.abs(
        alpha - check["alpha"]
    ).max()

    assert torch.equal(net.premium(-muhat, tauhat), net.premium(muhat, tauhat))
    assert np.allclose(net.policy(-muhat, tauhat).numpy(), 1.0 - alpha)

    _, _, l_np = NumpyTwoArm().learning(check["muhat"], check["tauhat"])
    gap = np.abs(l_ab - l_np) / np.maximum(np.abs(l_ab), 1e-3)
    assert np.quantile(gap, 0.99) < 1e-5, np.quantile(gap, 0.99)
    alpha_np = NumpyTwoArm().policy(check["muhat"], check["tauhat"])
    assert np.abs(alpha - alpha_np).max() < 1e-3, np.abs(alpha - alpha_np).max()
    print(f"two_arm on torch: matches pinn and numpy on {len(muhat)} states")
