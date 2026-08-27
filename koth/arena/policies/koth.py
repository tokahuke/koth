"""The package itself as an arena entrant, on the torch backend."""

from __future__ import annotations

import torch

from torch import Tensor
from typing import ClassVar, Self, override

from ... import Test
from ...torch_.decide import Decider
from ...torch_.state import State
from ...two_arm import PRIOR_FLOOR
from ..harness import Bayesian, Params, UnsupportedNumberOfArms


class Koth(Bayesian):
    """
    `Test.decide` on the filter's posterior every epoch, rate `rho = 1 - gamma`.
    The prior is the flattest the nets were trained at: every arm at precision
    `2 PRIOR_FLOOR / (rho sigma**2)`, so every contrast starts at
    `tauhat = PRIOR_FLOOR`. Subclasses fix `K`.
    """

    K: ClassVar[int]
    """Contenders per epoch, 2 or 3."""

    def __init__(
        self,
        params: Params,
        reps: int,
        device: str,
        sigma: float | None = None,
        eta: float | None = None,
    ) -> None:
        if params.arms < self.K:
            raise UnsupportedNumberOfArms(self.K, params.arms)
        rho = 1.0 - params.gamma
        sigma = params.sigma if sigma is None else sigma
        super().__init__(
            params,
            reps,
            device,
            sigma,
            eta,
            prior_precision=2.0 * PRIOR_FLOOR / (rho * sigma**2),
        )
        self.rho = rho
        self.test = Test(rho=rho, sigma=self.sigma, backend=Decider())

    @classmethod
    @override
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(params, reps, device, sigma=cls.SIGMA_FACTOR * params.sigma)

    @override
    def propose(self) -> Tensor:
        state = State(self.mean, torch.diag_embed(1.0 / self.precision))

        return self.test.decide(state, self.K).allocation.float()


class Koth2(Koth):
    """Pairs: the two-arm net on the most promising pair."""

    K = 2


class Koth3(Koth):
    """Triples: the three-arm net on the most promising triple."""

    K = 3


if __name__ == "__main__":
    from ..harness import Normal

    params = Params(
        gamma=0.999, horizon=500, sigma=1.0, effect=0.0, effect_std=0.3, size=1, arms=3
    )
    two_arms = Params(**{**params.__dict__, "arms": 2})
    eight_arms = Params(**{**params.__dict__, "arms": 8})

    # At k = arms the entrant is the net on the filter's contrasts.
    policy = Koth2.init(two_arms, 4, "cpu")
    policy.observe(
        (
            torch.tensor([[0.0, 0.5]] * 4, dtype=torch.float64),
            torch.full((4, 2), 0.5, dtype=torch.float64),
        )
    )
    mean, covariance = policy.contrasts()
    share = policy.test.decider.nets[2].policy(
        mean[:, 0] / (policy.rho**0.5 * policy.sigma),
        policy.rho * policy.sigma**2 / covariance[:, 0, 0],
    )
    assert torch.allclose(policy.propose()[:, 1].double(), share, atol=1e-6)

    # Eight arms, one far ahead: both commit to it, and Koth3 refuses two arms.
    wide = torch.tensor([[0.0, -0.2, 1.5, -0.1, 0.0, -0.3, 0.1, -0.4]])

    for cls in (Koth2, Koth3):
        run = (
            Normal(eight_arms, [0]).run(cls.init(eight_arms, 1, "cpu"), wide).runs()[0]
        )
        assert run.committed == 2, (cls.__name__, run.committed, run.final_allocation)
        print(
            f"{cls.__name__} at 8 arms: regret {run.regret:.2f}, committed at {run.committed_at}"
        )

    try:
        Koth3.init(two_arms, 1, "cpu")

        raise AssertionError("must refuse two arms")
    except UnsupportedNumberOfArms:
        pass
