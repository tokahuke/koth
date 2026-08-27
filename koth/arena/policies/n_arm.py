"""The closed-form baselines, any arm count."""

from __future__ import annotations

import math
import numpy as np
import torch

from functools import cache
from torch import Tensor
from typing import Self, override

from ..harness import Bayesian, Params, optimal_deadline

_HERMITE_NODES = 24
"""Gauss-Hermite nodes for probability matching's one-dimensional integral per arm."""


class ExploreThenCommit(Bayesian):
    """Uniform to the deadline, then the posterior leader, however uncertain."""

    def __init__(
        self, params: Params, reps: int, device: str, sigma: float, deadline: int
    ) -> None:
        super().__init__(params, reps, device, sigma)
        self.deadline = deadline

    @classmethod
    @override
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(
            params,
            reps,
            device,
            sigma=cls.SIGMA_FACTOR * params.sigma,
            deadline=optimal_deadline(params.gamma, params.horizon),
        )

    @override
    def propose(self) -> Tensor:
        if self.count < self.deadline:
            return self.uniform()

        return self.leader()


@cache
def _hermite(device: torch.device) -> tuple[Tensor, Tensor]:
    """Probabilists' Gauss-Hermite nodes and weights (weights sum to 1)."""
    x, w = np.polynomial.hermite_e.hermegauss(_HERMITE_NODES)

    return (
        torch.tensor(x, dtype=torch.float64, device=device),
        torch.tensor(w / w.sum(), dtype=torch.float64, device=device),
    )


class ProbabilityMatching(Bayesian):
    """
    Thompson as an allocation: the posterior probability of each arm being best,
    `P(i best) = int phi(z) prod_j Phi((m_i - m_j + s_i z) / s_j) dz`, one
    Gauss-Hermite integral per arm; uniform until every arm has data. It reaches a
    vertex (and so commits) only where the CDFs saturate in float64.
    """

    @override
    def propose(self) -> Tensor:
        nodes, weights = _hermite(self.mean.device)
        mean, sd = self.mean, self.deviation
        # (reps, i, j, node): arm i at its node value against every arm j.
        z_i = mean[:, :, None, None] + sd[:, :, None, None] * nodes
        cdf = torch.special.ndtr((z_i - mean[:, None, :, None]) / sd[:, None, :, None])
        eye = torch.eye(self.arms, dtype=torch.bool, device=mean.device)
        product = cdf.masked_fill(eye[None, :, :, None], 1.0).prod(dim=2)
        allocation = (product * weights).sum(dim=2)
        allocation = (allocation / allocation.sum(dim=1, keepdim=True)).float()

        return torch.where(self.live.unsqueeze(1), allocation, self.uniform())


class Elimination(Bayesian):
    """
    Successive elimination: split evenly among survivors, an arm is out once any
    other is pairwise-significantly better at `p_value`, retested every epoch from
    the accumulated data (so the peeking caveat applies). At two arms this is the
    z-test.
    """

    # ponytail: fixed nominal level, no alpha spending. Swap `threshold` for an
    # O'Brien-Fleming or Pocock boundary in `count` if the peeking cost matters.
    p_value: float = 0.05

    @override
    def propose(self) -> Tensor:
        threshold = float(torch.special.ndtri(torch.tensor(1.0 - self.p_value / 2.0)))
        mean, variance = self.mean, self.deviation**2
        z = (mean.unsqueeze(2) - mean.unsqueeze(1)) / (
            variance.unsqueeze(2) + variance.unsqueeze(1)
        ).sqrt()
        survivors = (~(z < -threshold).any(dim=2)).float()
        count = survivors.sum(dim=1)
        empty = count < 0.5
        allocation = torch.where(
            empty.unsqueeze(1),
            self.leader(),
            survivors / count.masked_fill(empty, 1.0).unsqueeze(1),
        )

        return torch.where(self.live.unsqueeze(1), allocation, self.uniform())


def _brezzi_lai(s: Tensor) -> Tensor:
    """
    Brezzi & Lai (2002) closed-form approximation of the normal Gittins index's
    boundary, `psi(s)` in units of posterior sd, `s = 1 / (n c)` for `n` effective
    observations and discount rate `c = -log(gamma)`.
    """
    root = s.clamp_min(1e-300).sqrt()
    large = (
        (2.0 * s.log() - s.log().clamp_min(1e-300).log() - math.log(16.0 * math.pi))
        .clamp_min(0.0)
        .sqrt()
    )

    return torch.where(
        s <= 0.2,
        (s / 2.0).sqrt(),
        torch.where(
            s <= 1.0,
            0.49 - 0.11 / root,
            torch.where(
                s <= 5.0,
                0.63 - 0.26 / root,
                torch.where(s <= 15.0, 0.77 - 0.58 / root, large),
            ),
        ),
    )


class Gittins(Bayesian):
    """
    The Gittins index policy, optimal for independent arms under geometric
    discounting: play the arm whose index `m_i + sd_i psi(sd_i^2 / (sigma^2 c))` is
    largest, `psi` the Brezzi-Lai approximation (within a few percent of the exact
    index) of `1 / (n c)`, `n = sigma^2 / var` the effective observations and
    `c = -log(gamma)`. An unsampled arm has infinite index, so every arm
    is tried once first. Plays a vertex every epoch, so the harness's commit
    record reads epoch 0 and means nothing here; read regret.
    """

    @override
    def propose(self) -> Tensor:
        c = -math.log(self.params.gamma)
        variance = 1.0 / self.precision
        index = self.mean + variance.sqrt() * _brezzi_lai(
            variance / (self.sigma**2 * c)
        )
        index = index.masked_fill(self.precision == 0.0, math.inf)

        return self.vertex(index.argmax(dim=1))


def demo() -> None:
    """Each baseline reaches the border its docstring claims, at two and eight arms."""
    from ..harness import Normal, Policy, Run

    def play(cls: type[Policy], deltas: Tensor) -> Run:
        """One rep of `cls` against fixed true effects."""
        params = Params(
            gamma=0.999,
            horizon=500,
            sigma=1.0,
            effect=0.0,
            effect_std=0.3,
            size=1,
            arms=deltas.shape[1],
        )
        world = Normal(params, [0])

        return world.run(cls.init(params, 1, "cpu"), deltas).runs()[0]

    winner, loser = torch.tensor([[0.0, 0.5]]), torch.tensor([[0.0, -0.5]])
    assert play(ProbabilityMatching, winner).final_allocation[1] > 0.95
    assert play(ProbabilityMatching, loser).final_allocation[1] < 0.1
    assert abs(play(ProbabilityMatching, torch.zeros(1, 2)).regret) < 1e-9

    etc = play(ExploreThenCommit, winner)
    assert etc.committed == 1 and etc.committed_at == optimal_deadline(0.999, 500)
    # Soft commit time is the commit epoch: 50/50 is worth 1, a vertex 0.
    assert abs(etc.precision_time - etc.committed_at) < 1e-6, etc.precision_time
    # Off-best allocation: half of every exploring epoch at two arms, discounted,
    # none after committing to the winner.
    explored = 0.5 * (1.0 - 0.999**etc.committed_at) / (1.0 - 0.999)
    assert abs(etc.off_best - explored) < 1e-6, etc.off_best
    assert abs(play(ExploreThenCommit, loser).off_best - explored) < 1e-6
    assert play(Elimination, winner).committed == 1

    gittins = play(Gittins, winner)
    assert gittins.final_allocation[1] == 1.0 and gittins.regret < etc.regret
    # psi against Brezzi-Lai's own table points: psi(0.2) = 0.316, psi(1) = 0.38,
    # psi(5) = 0.514, psi(15) = 0.62 (their Table 1, rounded).
    psi = _brezzi_lai(torch.tensor([0.2, 1.0, 5.0, 15.0]))
    assert torch.allclose(psi, torch.tensor([0.316, 0.38, 0.514, 0.62]), atol=0.01), psi

    wide = torch.tensor([[0.0, -0.2, 1.5, -0.1, 0.0, -0.3, 0.1, -0.4]])
    assert play(Gittins, wide).final_allocation[2] == 1.0
    assert play(ExploreThenCommit, wide).committed == 2
    assert play(Elimination, wide).committed == 2
    assert play(ProbabilityMatching, wide).final_allocation[2] > 0.9

    # The filter under drift: precision stops at the recursion's fixed point,
    # p/2 + sqrt(p^2/4 + p/eta^2) for a lump p per epoch.
    params = Params(
        gamma=0.999, horizon=1, sigma=1.0, effect=0.0, effect_std=0.0, size=1, arms=2
    )
    aware = ProbabilityMatching(params, 1, "cpu", sigma=1.0, eta=0.05)
    lump = torch.full((1, 2), 0.25, dtype=torch.float64)
    for _ in range(400):
        aware.observe((torch.full((1, 2), 0.5, dtype=torch.float64), lump))
    fixed_point = 0.25 / 2.0 + math.sqrt(0.25**2 / 4.0 + 0.25 / 0.05**2)
    assert abs(float(aware.precision[0, 0]) - fixed_point) < 1e-6

    print(f"etc regret {etc.regret:.2f}, committed at epoch {etc.committed_at}")
    print(f"drift fixed point {fixed_point:.2f}")


if __name__ == "__main__":
    demo()
