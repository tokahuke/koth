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


class KotH(Bayesian):
    """
    `Test.decide` every epoch on KotH's own belief, a full-covariance `State`,
    rate `rho = 1 - gamma`: a reading on one arm moves every arm the prior ties
    to it. Flat start: every arm at the flattest precision the nets were trained
    at, `2 PRIOR_FLOOR / (rho sigma**2)`, so every contrast starts at
    `tauhat = PRIOR_FLOOR`; `prime` starts from a world's law instead, whole.
    Drift is the same forgetting step the arena filter takes. Subclasses fix `K`.
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
        self.state = State(self.mean, torch.diag_embed(1.0 / self.precision))
        decider = Decider()

        # The nets' weights live where the filter's tensors do.
        for net in decider.nets.values():
            net.w = {name: value.to(device) for name, value in net.w.items()}
        self.test = Test(rho=rho, sigma=self.sigma, backend=decider)

    @classmethod
    @override
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(
            params,
            reps,
            device,
            sigma=cls.SIGMA_FACTOR * params.sigma,
            eta=cls.ETA_FACTOR * params.eta,
        )

    @override
    def prime(self, mean: Tensor, cov: Tensor) -> None:
        """The whole law, correlation included."""
        super().prime(mean, cov)
        self.state = State(
            mean.double().expand(self.reps, -1).clone(),
            (
                cov.double()
                + 1e-12 * torch.eye(self.arms, dtype=torch.float64, device=cov.device)
            )
            .expand(self.reps, -1, -1)
            .clone(),
        )

    @override
    def observe(self, observation: tuple[Tensor, Tensor]) -> None:
        super().observe(observation)
        estimate, design = observation
        # The world walks at eta between readings: every arm's variance grows.
        walked = self.state.cov + self.eta**2 * torch.eye(
            self.arms, dtype=self.state.cov.dtype, device=self.state.cov.device
        )
        self.state = State(self.state.mean, walked).update(
            estimate.double(), design.double() / self.sigma**2
        )

    @override
    def propose(self) -> Tensor:
        return self.test.decide(self.state, self.K).allocation.float()


_DRAWS = 4096
"""Standard-normal draws behind `JointThompson`'s P(best), shared by every rep and
epoch (common random numbers): the estimate is a deterministic function of the
posterior, so reps stay independent of their batch."""


class JointThompson(Bayesian):
    """
    Thompson as an allocation on a *joint* posterior: KotH's full-covariance
    `State`, so a primed correlation carries evidence across arms, and
    `P(i best)` by Monte Carlo over the joint (no closed form once arms
    correlate). Uniform until every arm has data; flat, it is
    `ProbabilityMatching` up to Monte Carlo error.
    """

    def __init__(
        self,
        params: Params,
        reps: int,
        device: str,
        sigma: float | None = None,
        eta: float | None = None,
    ) -> None:
        super().__init__(params, reps, device, sigma, eta)
        self.state: State | None = None
        generator = torch.Generator().manual_seed(2**20)
        self.draws = torch.randn(
            _DRAWS, params.arms, generator=generator, dtype=torch.float64
        ).to(device)

    @override
    def prime(self, mean: Tensor, cov: Tensor) -> None:
        """The whole law, correlation included."""
        super().prime(mean, cov)
        self.state = State(
            mean.double().expand(self.reps, -1).clone(),
            (
                cov.double()
                + 1e-12 * torch.eye(self.arms, dtype=torch.float64, device=cov.device)
            )
            .expand(self.reps, -1, -1)
            .clone(),
        )

    @override
    def observe(self, observation: tuple[Tensor, Tensor]) -> None:
        super().observe(observation)

        if self.state is None:
            return
        estimate, design = observation
        walked = self.state.cov + self.eta**2 * torch.eye(
            self.arms, dtype=self.state.cov.dtype, device=self.state.cov.device
        )
        self.state = State(self.state.mean, walked).update(
            estimate.double(), design.double() / self.sigma**2
        )

    @override
    def propose(self) -> Tensor:
        if self.state is None:
            # Flat: the per-arm posterior is the joint one.
            mean, cov = self.mean, torch.diag_embed(1.0 / self.precision)
        else:
            mean, cov = self.state.mean, self.state.cov
        root = torch.linalg.cholesky(cov)
        samples = mean[:, None, :] + self.draws[None] @ root.transpose(1, 2)
        best = samples.argmax(-1)
        counts = torch.zeros_like(mean).scatter_add_(
            1, best, torch.ones_like(best, dtype=mean.dtype)
        )
        allocation = (counts / _DRAWS).float()

        return torch.where(self.live.unsqueeze(1), allocation, self.uniform())


class KotH2(KotH):
    """Pairs: the two-arm net on the most promising pair."""

    K = 2


class KotH3(KotH):
    """Triples: the three-arm net on the most promising triple."""

    K = 3


if __name__ == "__main__":
    from ..harness import Normal

    params = Params(
        gamma=0.999, horizon=500, sigma=1.0, effect=0.0, effect_std=0.3, arms=3
    )
    two_arms = Params(**{**params.__dict__, "arms": 2})
    eight_arms = Params(**{**params.__dict__, "arms": 8})

    # At k = arms the entrant is the net on the filter's contrasts.
    policy = KotH2.init(two_arms, 4, "cpu")
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

    # Eight arms, one far ahead: both commit to it, and KotH3 refuses two arms.
    wide = torch.tensor([[0.0, -0.2, 1.5, -0.1, 0.0, -0.3, 0.1, -0.4]])

    for cls in (KotH2, KotH3):
        run = (
            Normal(eight_arms, [0]).run(cls.init(eight_arms, 1, "cpu"), wide).runs()[0]
        )
        assert run.committed == 2, (cls.__name__, run.committed, run.final_allocation)
        print(
            f"{cls.__name__} at 8 arms: regret {run.regret:.2f}, committed at {run.committed_at}"
        )

    try:
        KotH3.init(two_arms, 1, "cpu")

        raise AssertionError("must refuse two arms")
    except UnsupportedNumberOfArms:
        pass

    # Primed with a kernel prior, a reading on bid 2 moves the belief on bid 3
    # and leaves bid 5 nearly alone; flat, nothing else moves.
    from ..harness import Matern
    from .n_arm import ProbabilityMatching

    ladder = Params(
        gamma=0.999, horizon=10, sigma=1.0, effect=0.0, effect_std=1.0, arms=6
    )
    world = Matern(ladder, [0], lengthscale=1.5)
    primed = KotH3.init(ladder, 1, "cpu")
    primed.prime(*world.prior())
    flat = KotH3.init(ladder, 1, "cpu")
    reading = (
        torch.tensor([[0.0, 0.0, 3.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
    )
    primed.observe(reading)
    flat.observe(reading)
    assert primed.state.mean[0, 2] > 1.0 and primed.state.mean[0, 3] > 0.5
    assert primed.state.mean[0, 3] > 5 * primed.state.mean[0, 5]
    assert flat.state.mean[0, 3] == 0.0 and flat.state.mean[0, 2] > 1.0
    print("KotH primed: a reading on one bid moves its neighbour")

    # JointThompson: flat, it is probability matching up to Monte Carlo error;
    # primed on the ladder, a reading on bid 2 raises bid 3's share.
    flat_joint = JointThompson.init(ladder, 1, "cpu")
    matched = ProbabilityMatching.init(ladder, 1, "cpu")
    sharp = (
        torch.tensor([[0.0, 0.5, 1.0, 0.2, -0.3, 0.1]], dtype=torch.float64),
        torch.full((1, 6), 1.0, dtype=torch.float64),
    )

    for _ in range(3):
        flat_joint.observe(sharp)
        matched.observe(sharp)
    assert (flat_joint.propose() - matched.propose()).abs().max() < 0.03
    primed_joint = JointThompson.init(ladder, 1, "cpu")
    primed_joint.prime(*world.prior())
    before = primed_joint.propose()[0, 3]
    primed_joint.observe(reading)
    assert primed_joint.propose()[0, 3] > before
    print("joint thompson: matches probability matching flat, uses the prior primed")
