"""The two-arm net on numpy, derivatives by central differences."""

from __future__ import annotations

import math
import numpy as np

from scipy import special
from typing import override

from .. import two_arm

STEP = 1e-4
"""
Central-difference step on the chart, float64: second-derivative error ~1e-8
in the bulk. Across the free boundary the premium is C^1 only, so differences
smear its curvature jump over a band this wide; there `L_ab` is off by
O(1e-2) and the policy is unmoved, the stationary point sitting far outside
[0, 1] on both readings.
"""


def nu(mean: np.ndarray, stddev: np.ndarray) -> np.ndarray:
    """E[max(0, X)] for X ~ N(mean, stddev**2), the free-information envelope."""
    z = mean / stddev
    expectation = mean * special.ndtr(z) + stddev * np.exp(-0.5 * z**2) / math.sqrt(
        2.0 * math.pi
    )

    return np.maximum(expectation, 0.0)


class TwoArm(two_arm.TwoArm[np.ndarray]):
    """The two-arm net on numpy."""

    def __init__(self) -> None:
        self.w = two_arm.weights()

    @override
    def premium(self, muhat: np.ndarray, tauhat: np.ndarray) -> np.ndarray:
        return self._premium(np.abs(muhat), tauhat)

    @override
    def _premium(self, muhat: np.ndarray, tauhat: np.ndarray) -> np.ndarray:
        """The net as trained, `muhat >= 0` only."""
        muhat, tauhat = np.broadcast_arrays(muhat, tauhat)
        tau_eff = np.maximum(tauhat, two_arm.PRIOR_FLOOR)
        mu_eff = muhat * np.sqrt(tauhat / tau_eff)
        x = (
            np.stack(
                [mu_eff, np.log(tau_eff), mu_eff * np.sqrt(tau_eff), mu_eff * tau_eff],
                -1,
            )
            / self.w["feature_scale"]
        )
        layer = 0

        while f"g{layer}" in self.w:
            x = np.tanh(
                self.w[f"g{layer}"] * (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])
            )
            layer += 1
        response = (x @ self.w[f"w{layer}"].T + self.w[f"b{layer}"])[..., 0]
        y = np.maximum(response, 0.0) ** 2

        return (
            nu(-muhat, 1.0 / np.sqrt(tauhat))
            * np.exp(self.w["log_scale"])
            * y
            / (1.0 + y)
        )

    @override
    def value(self, muhat: np.ndarray, tauhat: np.ndarray) -> np.ndarray:
        return np.maximum(muhat, 0.0) + self.premium(muhat, tauhat)

    @override
    def _chart(self, z: np.ndarray, s: np.ndarray) -> np.ndarray:
        """The premium on the similarity chart, `g = e^(s/2) u(z e^(-s/2), e^s)`."""
        return np.exp(s / 2.0) * self._premium(z * np.exp(-s / 2.0), np.exp(s))

    @override
    def learning(
        self, muhat: np.ndarray, tauhat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z, s = np.abs(muhat) * np.sqrt(tauhat), np.log(tauhat)
        h = STEP * np.maximum(1.0, np.abs(z))
        g = self._chart(z, s)
        up, down = self._chart(z + h, s), self._chart(z - h, s)
        g_z, g_zz = (up - down) / (2.0 * h), (up - 2.0 * g + down) / h**2
        g_s = (self._chart(z, s + STEP) - self._chart(z, s - STEP)) / (2.0 * STEP)

        return z, s, g_s + 0.5 * g_zz + 0.5 * z * g_z - 0.5 * g

    @override
    def policy(self, muhat: np.ndarray, tauhat: np.ndarray) -> np.ndarray:
        z, s, l_ab = self.learning(muhat, tauhat)
        c_xx, c_x = -l_ab, np.exp(s) * z + l_ab
        safe = np.where(np.abs(c_xx) < 1e-12, 1.0, c_xx)
        vertex = np.clip(-c_x / (2.0 * safe), 0.0, 1.0)
        candidates = np.stack([vertex, np.zeros_like(vertex), np.ones_like(vertex)], -1)
        values = c_xx[..., None] * candidates**2 + c_x[..., None] * candidates
        share = np.take_along_axis(candidates, values.argmax(-1)[..., None], -1)[..., 0]

        return np.where(muhat >= 0.0, share, 1.0 - share)

    @override
    def subset_value(self, contrast: np.ndarray, precision: np.ndarray) -> np.ndarray:
        return self.value(contrast[..., 0], precision[..., 0, 0])

    @override
    def subset_policy(self, contrast: np.ndarray, precision: np.ndarray) -> np.ndarray:
        share = self.policy(contrast[..., 0], precision[..., 0, 0])

        return np.stack([1.0 - share, share], -1)


if __name__ == "__main__":
    check = two_arm.fixture()
    muhat, tauhat = check["muhat"], check["tauhat"]
    net = TwoArm()

    # The premium against pinn's float32 evaluation of the same net.
    u = net.premium(muhat, tauhat)
    assert np.allclose(u, check["u"], rtol=1e-4, atol=1e-6), np.abs(
        u - check["u"]
    ).max()

    # The learning number by central differences against autograd, and the
    # policy it reads: off by the float32 wobble and the boundary band, no more.
    _, _, l_ab = net.learning(muhat, tauhat)
    gap = np.abs(l_ab - check["l_ab"]) / np.maximum(np.abs(check["l_ab"]), 1e-3)
    assert np.median(gap) < 1e-4 and np.quantile(gap, 0.99) < 1e-2, (
        np.median(gap),
        np.quantile(gap, 0.99),
    )
    alpha = net.policy(muhat, tauhat)
    assert np.abs(alpha - check["alpha"]).max() < 1e-3, np.abs(
        alpha - check["alpha"]
    ).max()
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()

    # Either sign: the premium is even and the policy reflects.
    assert np.array_equal(net.premium(-muhat, tauhat), u)
    assert np.allclose(net.policy(-muhat, tauhat), 1.0 - alpha)

    # Batched shapes broadcast; a scalar state works.
    assert net.policy(muhat.reshape(40, 50), tauhat.reshape(40, 50)).shape == (40, 50)
    assert 0.0 <= float(net.policy(np.float64(0.5), np.float64(1.0))) <= 1.0
    print(
        f"two_arm on numpy: premium, learning and policy match pinn on {len(muhat)} states"
    )
