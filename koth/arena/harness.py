"""
The arena's core: the environment (an ABC with the simulation loop, `Normal` its
one implementation), the policy contract, the generic Bayesian filter every strategy
inherits, and discounted-regret bookkeeping. N-arm throughout: allocations are
simplex rows, arm 0 is the control and starts at effect 0.

Everything is vectorized over *reps*: state tensors carry a leading (reps,)
dimension, the epoch loop is the only sequential axis, and each rep's noise stream is
a function of its seed alone, so a rep's numbers do not depend on the batch around it
(the demo asserts it).
"""

from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from torch import Tensor
from typing import ClassVar, Self, override


@dataclass
class Params:
    """One environment: what the world does, and how long and how hard it is played."""

    gamma: float
    """Discount factor per epoch, so regret at epoch t is weighted gamma**t."""

    horizon: int
    """Epochs simulated, every one of them, whether or not a policy has committed."""

    sigma: float
    """Noise scale of one observation."""

    effect: float
    """
    Mean of the environment's draw of the true effects. No policy reads it: priors are
    *policy parameters*, not environment knowledge, and matching one to the other is a
    separate study.
    """

    effect_std: float
    """Spread of that draw, read by the environment alone for the same reason."""

    size: int
    """Reps in the sweep."""

    arms: int
    """Arm count, control included; a strategy refuses a count it cannot play."""

    eta: float = 0.0
    """
    Drift volatility of every arm's effect per epoch (a two-arm contrast therefore
    walks at sqrt(2) eta). 0 is a static world, and no branch anywhere depends on it.
    This is the *environment's* eta; a policy's belief about it is a policy parameter.
    """


class UnsupportedNumberOfArms(ValueError):
    """Raised by `Policy.init` for an arm count the policy cannot play.
    The sweep skips the policy."""

    def __init__(self, expected: int, got: int) -> None:
        super().__init__(f"plays {expected} arms, asked for {got}")
        self.expected = expected
        self.got = got


class Policy(ABC):
    """
    A policy over a batch of reps: state is (reps, ...)-shaped, `propose` returns
    (reps, arms), `observe` takes the batched observation. Rep i's numbers must not
    depend on the batch around it, which the demo asserts.
    """

    @classmethod
    @abstractmethod
    def init(cls, params: Params, reps: int, device: str) -> Self:
        """The policy as the sweep builds it, with every parameter tied to `params`."""

    @abstractmethod
    def observe(self, observation: object) -> None:
        """Fold one epoch's evidence in. Its shape is the environment's own."""

    @abstractmethod
    def propose(self) -> Tensor:
        """This epoch's allocation, one simplex row per rep."""


def optimal_deadline(gamma: float, horizon: int) -> int:
    """
    Deadline for ExploreThenCommit: T**(2/3) with a leading constant of 1, T the
    shorter of the discount's 1/rho and the hard horizon. It reads the horizon and
    the discount and nothing else, deliberately.
    """
    effective = min(float(horizon), 1.0 / (1.0 - gamma))

    return max(1, round(effective ** (2.0 / 3.0)))


@dataclass
class Run:
    """
    One simulated experiment: what was allocated and what it cost.

    `regret` is discounted at the environment's gamma, so it compares across policies
    only within one environment. `delta` is the effect at epoch 0, which under drift
    is the starting point rather than the truth throughout. `committed` and
    `committed_at` are a record, not a stopping condition: drift makes a vertex
    escapable.
    """

    delta: list[float]
    policy: str = ""
    epochs: int = 0
    precision_time: float = 0.0
    final_allocation: list[float] = field(default_factory=list)
    regret: float = 0.0
    committed: int | None = None
    committed_at: int | None = None
    off_best: float = 0.0
    """
    Allocation sent to arms other than the epoch's true best, discounted at the
    environment's gamma like `regret`: a vertex on a loser counts one epoch, an
    even split of N arms (N - 1) / N, a vertex on the winner zero. Regret with
    unit gaps, which counts an index policy's switching where `precision_time`
    cannot.
    """


@dataclass
class Batch:
    """
    `Run` over a whole batch: the same bookkeeping, each field a (reps,)- or
    (reps, arms)-shaped tensor, `committed`/`committed_at` carrying -1 for
    never. `runs()` explodes back to scalar Runs, the pickle format analyze
    reads.
    """

    delta: Tensor
    regret: Tensor
    precision_time: Tensor
    off_best: Tensor
    final_allocation: Tensor
    committed: Tensor
    committed_at: Tensor
    epochs: int
    policy: str

    def runs(self) -> list[Run]:
        """One scalar `Run` per rep, which is the pickle format `analyze` reads."""
        delta = self.delta.cpu()
        regret = self.regret.cpu()
        precision_time = self.precision_time.cpu()
        off_best = self.off_best.cpu()
        final_allocation = self.final_allocation.cpu()
        committed = self.committed.cpu()
        committed_at = self.committed_at.cpu()
        out = []

        for i in range(delta.shape[0]):
            at = int(committed_at[i])
            out.append(
                Run(
                    delta=[float(d) for d in delta[i]],
                    policy=self.policy,
                    epochs=self.epochs,
                    precision_time=float(precision_time[i]),
                    final_allocation=[float(a) for a in final_allocation[i]],
                    regret=float(regret[i]),
                    committed=int(committed[i]) if at >= 0 else None,
                    committed_at=at if at >= 0 else None,
                    off_best=float(off_best[i]),
                )
            )

        return out


@dataclass
class Study:
    """
    A whole simulation as pickled: the environment it ran under, plus every run.

    Params travels with the runs because regret is discounted at that gamma, so the
    numbers are meaningless without it.
    """

    params: Params
    runs: list[Run]


class Environment(ABC):
    """
    The world a batch of reps is played against: the truth, how it moves between
    epochs, and what an allocation reveals about it. One noise row per seed, pre-drawn
    and consumed through per-rep cursors, so a rep's stream is independent of its
    batch; a masked `normal` advances only the consuming reps' cursors.
    """

    def __init__(self, params: Params, seeds: list[int], device: str = "cpu") -> None:
        self.params = params
        self.seeds = seeds
        self.device = device
        capacity = self.effect_draws + self.draws_per_epoch * params.horizon
        rows = torch.empty(len(seeds), capacity)

        for i, seed in enumerate(seeds):
            generator = torch.Generator()
            generator.manual_seed(seed)
            rows[i] = torch.randn(capacity, generator=generator)

        self.noise = rows.to(device)
        self.cursor = torch.zeros(len(seeds), dtype=torch.long, device=device)

    @property
    @abstractmethod
    def draws_per_epoch(self) -> int:
        """Variates one rep consumes per epoch."""

    @property
    @abstractmethod
    def effect_draws(self) -> int:
        """Variates one rep consumes for the effect draw."""

    @abstractmethod
    def draw_effect(self) -> Tensor:
        """The truth at epoch 0, (reps, arms), arm 0 at 0."""

    @abstractmethod
    def advance(self, deltas: Tensor) -> Tensor:
        """The truth one epoch later."""

    @abstractmethod
    def observe(self, allocation: Tensor, deltas: Tensor) -> object:
        """What one epoch at `allocation` reveals; the policies' `observe` reads it."""

    def normal(
        self,
        mean: Tensor | float,
        deviation: Tensor | float,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        The next variate of every rep's stream, scaled per rep; masked-out reps consume
        nothing and read 0.
        """
        draw = self.noise.gather(1, self.cursor.unsqueeze(1)).squeeze(1).double()

        if mask is None:
            self.cursor += 1

            return draw * deviation + mean

        self.cursor += mask.long()

        return torch.where(mask, draw * deviation + mean, torch.zeros_like(draw))

    def run(
        self,
        policy: Policy,
        deltas: Tensor,
        progress: Callable[[int], None] | None = None,
    ) -> Batch:
        """
        Play `policy` for the full horizon, all reps at once; the epoch loop is the only
        sequential axis, commit detection a masked first crossing of an exact vertex.
        Regret is measured every epoch against an oracle on the best arm *at that
        epoch*. `progress`, if given, hears the epoch just finished.
        """
        reps, arms = deltas.shape
        start = deltas
        regret = torch.zeros(reps, dtype=torch.float64, device=self.device)
        precision_time = torch.zeros(reps, dtype=torch.float64, device=self.device)
        off_best = torch.zeros(reps, dtype=torch.float64, device=self.device)
        committed = torch.full((reps,), -1, dtype=torch.long, device=self.device)
        committed_at = torch.full((reps,), -1, dtype=torch.long, device=self.device)

        for epoch in range(self.params.horizon):
            allocation = policy.propose()
            best, best_arm = deltas.max(dim=1)
            reward = (allocation * deltas).sum(dim=1).double()
            regret += self.params.gamma**epoch * (best.double() - reward)
            off_best += self.params.gamma**epoch * (
                1.0 - allocation.gather(1, best_arm[:, None])[:, 0].double()
            )

            # Soft commit time: N (1 - sum a^2) / (N - 1) is 4a(1-a) at two arms, 1 at
            # uniform, 0 at a vertex. Summed, the uniform-equivalent epochs of evidence
            # bought.
            precision_time += (
                arms * (1.0 - (allocation**2).sum(dim=1).double()) / (arms - 1)
            )

            top, arm = allocation.max(dim=1)
            crossing = (top >= 1.0) & (committed_at < 0)
            committed = torch.where(crossing, arm, committed)
            committed_at = torch.where(crossing, epoch, committed_at)

            policy.observe(self.observe(allocation, deltas))
            deltas = self.advance(deltas)

            if progress is not None:
                progress(epoch + 1)

        return Batch(
            delta=start,
            regret=regret,
            precision_time=precision_time,
            off_best=off_best,
            final_allocation=allocation,
            committed=committed,
            committed_at=committed_at,
            epochs=self.params.horizon,
            policy=type(policy).__name__,
        )


class Normal(Environment):
    """
    Gaussian arms: challengers drawn iid from the study's distribution, every arm
    walking at `eta` between epochs (unconditionally, so nothing branches on drift),
    and an epoch at allocation `a` yielding per arm an estimate at noise
    `sigma / sqrt(a_i)`: the design, not a precision, because sigma is the
    policy's to believe. An unplayed arm consumes no draw and reads 0.
    """

    @property
    @override
    def draws_per_epoch(self) -> int:
        return 2 * self.params.arms

    @property
    @override
    def effect_draws(self) -> int:
        return self.params.arms - 1

    @override
    def draw_effect(self) -> Tensor:
        params = self.params
        columns = [
            self.normal(params.effect, params.effect_std)
            for _ in range(params.arms - 1)
        ]

        return torch.stack([torch.zeros_like(columns[0]), *columns], dim=1).float()

    @override
    def advance(self, deltas: Tensor) -> Tensor:
        steps = [self.normal(0.0, self.params.eta) for _ in range(self.params.arms)]

        return (deltas.double() + torch.stack(steps, dim=1)).float()

    @override
    def observe(self, allocation: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        design = allocation.double()
        live = design > 0.0
        deviation = self.params.sigma / design.masked_fill(~live, 1.0).sqrt()
        estimate = torch.stack(
            [
                self.normal(deltas[:, i].double(), deviation[:, i], live[:, i])
                for i in range(design.shape[1])
            ],
            dim=1,
        )

        return estimate, design


class Bayesian(Policy):
    """
    The filter every strategy inherits: an independent Kalman posterior per arm,
    (reps, arms) `mean` and `precision`, forecast-then-update. Under `eta = 0` and a
    flat prior it is the conjugate normal posterior; with drift, precision erodes as
    `p / (1 + eta^2 p)` before each update, so a flat start needs no special case.
    `sigma` and `eta` are the *policy's* beliefs: `init` sets `sigma` to
    `SIGMA_FACTOR` times the truth, so a misspecified contestant is a subclass
    with that attribute; `prior_precision` is per arm, 0 flat.
    """

    SIGMA_FACTOR: ClassVar[float] = 1.0
    """What the policy believes `sigma` is, as a multiple of the truth."""

    def __init__(
        self,
        params: Params,
        reps: int,
        device: str,
        sigma: float | None = None,
        eta: float | None = None,
        prior_precision: float = 0.0,
    ) -> None:
        self.params = params
        self.arms = params.arms
        self.sigma = params.sigma if sigma is None else sigma
        self.eta = params.eta if eta is None else eta
        self.count = 0
        self.mean = torch.zeros(reps, self.arms, dtype=torch.float64, device=device)
        self.precision = torch.full_like(self.mean, prior_precision)

    @classmethod
    @override
    def init(cls, params: Params, reps: int, device: str) -> Self:
        return cls(params, reps, device, sigma=cls.SIGMA_FACTOR * params.sigma)

    @override
    def observe(self, observation: tuple[Tensor, Tensor]) -> None:
        estimate, design = observation
        self.count += 1
        forecast = self.precision / (1.0 + self.eta**2 * self.precision)
        gained = design / self.sigma**2
        total = forecast + gained
        self.mean = torch.where(
            total > 0.0,
            (self.mean * forecast + estimate * gained)
            / total.masked_fill(total == 0.0, 1.0),
            self.mean,
        )
        self.precision = total

    @property
    def reps(self) -> int:
        """Reps in the batch."""
        return self.mean.shape[0]

    @property
    def live(self) -> Tensor:
        """Reps whose every arm has a proper posterior."""
        return (self.precision > 0.0).all(dim=1)

    @property
    def deviation(self) -> Tensor:
        """Posterior sd per arm, 1 where the posterior is still improper."""
        return self.precision.masked_fill(self.precision == 0.0, 1.0).rsqrt()

    def uniform(self) -> Tensor:
        """The even split, one row per rep."""
        return torch.full(
            (self.reps, self.arms), 1.0 / self.arms, device=self.mean.device
        )

    def vertex(self, arm: Tensor) -> Tensor:
        """An arm index per rep as a simplex row."""
        return torch.eye(self.arms, device=self.mean.device)[arm]

    def leader(self) -> Tensor:
        """First-maximal arm by posterior mean, as a vertex row."""
        return self.vertex(self.mean.argmax(dim=1))

    def contrasts(self, reference: int = 0) -> tuple[Tensor, Tensor]:
        """
        The posterior on the contrasts `theta_j - theta_reference`, j over the other
        arms in order: mean (reps, arms - 1) and covariance (reps, arms - 1, arms - 1),
        `diag(1 / p_j) + 1 / p_reference`. Improper posteriors give infinite entries.
        """
        others = [j for j in range(self.arms) if j != reference]
        variance = 1.0 / self.precision
        mean = self.mean[:, others] - self.mean[:, reference : reference + 1]
        covariance = (
            torch.diag_embed(variance[:, others]) + variance[:, reference, None, None]
        )

        return mean, covariance


def demo() -> None:
    """A rep's numbers do not depend on the batch around it, for every strategy."""
    from .policies import ALL

    params = Params(
        gamma=0.999,
        horizon=300,
        sigma=1.0,
        effect=0.3,
        effect_std=0.4,
        size=8,
        eta=0.02,
        arms=2,
    )
    reps = 8

    # Paired seeds survive batching: a permuted sub-batch reproduces its reps' numbers,
    # bitwise for the closed-form policies. The trajectory tolerances below cover the
    # float32 wobble.
    for arms in (2, 3, 5):
        params = replace(params, arms=arms)

        for cls in ALL:
            # ValueError, not the subclass: under `-m` this module is loaded twice
            # (as __main__ and as koth.arena.harness), with two copies of the class.
            try:
                policy = cls.init(params, reps, "cpu")
            except ValueError:
                continue

            world = Normal(params, list(range(reps)))
            batch = world.run(policy, world.draw_effect())

            seeds = [5, 2, 7]
            world = Normal(params, seeds)
            sub = world.run(cls.init(params, len(seeds), "cpu"), world.draw_effect())

            label = f"{cls.__name__} at {arms} arms"
            assert torch.equal(sub.committed_at, batch.committed_at[seeds]), label
            assert torch.equal(sub.committed, batch.committed[seeds]), label
            assert torch.equal(sub.delta, batch.delta[seeds]), label

            # A *net-carrying* policy is exempt from the trajectory comparisons: no
            # tolerance can be set for it, since the wobble is a property of the loaded
            # net. The exact fields above carry the test.
            if getattr(cls, "model_file", None) is not None:
                continue

            assert torch.allclose(
                sub.regret, batch.regret[seeds], rtol=1e-2, atol=1e-6
            ), label
            assert torch.allclose(
                sub.precision_time, batch.precision_time[seeds], rtol=1e-2, atol=1e-6
            ), label

        print(f"{arms} arms: sub-batch == full batch, rep for rep")


if __name__ == "__main__":
    demo()
