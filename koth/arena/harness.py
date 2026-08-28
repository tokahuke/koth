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
    world: str = "normal"
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
    world: str = "normal"

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
                    world=self.world,
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
    A whole simulation as pickled: every world it ran, by label, plus every run
    (each tagged with its world). The parameters travel with the runs because
    regret is discounted at that gamma, so the numbers mean nothing without them.
    """

    environments: dict[str, Params]
    runs: list[Run]


class Environment(ABC):
    """
    The world a batch of reps is played against: the truth, how it moves between
    epochs, and what an allocation reveals about it. One noise row per seed, pre-drawn
    and consumed through per-rep cursors, so a rep's stream is independent of its
    batch; a masked `normal` advances only the consuming reps' cursors.
    """

    @classmethod
    def describe(cls, fields: dict[str, object], options: dict[str, object]) -> Params:
        """
        The `Params` a world of this kind tells the policies, from the spec's
        fields and the kind's own options; a kind that fixes a field derives it here.
        """
        return Params(**fields)

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
    def prior(self) -> tuple[Tensor, Tensor]:
        """
        The law `draw_effect` draws from, as a Gaussian over the arms: mean
        `(arms,)` and covariance `(arms, arms)`. What a contestant with
        `PRIOR = "world"` starts from.
        """

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
    def prior(self) -> tuple[Tensor, Tensor]:
        arms = self.params.arms
        mean = torch.full((arms,), self.params.effect, dtype=torch.float64)
        variance = torch.full((arms,), self.params.effect_std**2, dtype=torch.float64)
        mean[0], variance[0] = 0.0, 0.0

        return mean.to(self.device), torch.diag(variance).to(self.device)

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


class Student(Normal):
    """
    `Normal` with Student-t observation noise at `df` degrees of freedom, scaled
    to the same variance `sigma**2` (so `df > 2`): the policies' Gaussian model
    is wrong in shape only, and the outliers get heavier as `df` falls. Effect
    draws and drift stay Gaussian. The t's denominators come from a second
    per-rep stream, so a rep's numbers still depend on its seed alone.
    """

    def __init__(
        self, params: Params, seeds: list[int], device: str = "cpu", df: float = 3.0
    ) -> None:
        if df <= 2.0:
            raise ValueError(f"df must exceed 2 for a finite variance, got {df}")
        super().__init__(params, seeds, device)
        self.df = df
        capacity = params.arms * params.horizon
        rows = torch.empty(len(seeds), capacity)

        for i, seed in enumerate(seeds):
            generator = torch.Generator()
            generator.manual_seed(seed + 2**31)
            rows[i] = (torch.randn(capacity, int(df), generator=generator) ** 2).sum(1)
        # chi-squared(df) / df, one per arm per epoch, consumed by epoch.
        self.denominators = (rows / df).to(device)
        self.epoch = 0

    @override
    def observe(self, allocation: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        estimate, design = super().observe(allocation, deltas)
        arms = design.shape[1]
        chi = self.denominators[:, self.epoch * arms : (self.epoch + 1) * arms].double()
        self.epoch += 1
        # t = z / sqrt(chi2 / df), times sqrt((df - 2) / df) for unit variance.
        factor = ((self.df - 2.0) / self.df) ** 0.5 / chi.sqrt()

        return deltas.double() + (estimate - deltas.double()) * factor, design


class AR1(Normal):
    """
    `Normal` with observation noise that remembers: each arm's standardized noise
    follows `n_t = phi n_(t-1) + sqrt(1 - phi**2) z_t`, so its marginal variance
    stays `sigma**2` and only the dependence between epochs changes; the
    policies' filter, which takes readings as independent, credits each one with
    more information than it carries. An arm that gets no reading in an epoch
    has its noise decay by `phi` (no fresh innovation, since no draw is consumed).
    """

    def __init__(
        self, params: Params, seeds: list[int], device: str = "cpu", phi: float = 0.5
    ) -> None:
        if not 0.0 <= phi < 1.0:
            raise ValueError(f"phi must be in [0, 1), got {phi}")
        super().__init__(params, seeds, device)
        self.phi = phi
        self.memory = torch.zeros(
            len(seeds), params.arms, dtype=torch.float64, device=device
        )

    @override
    def observe(self, allocation: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        estimate, design = super().observe(allocation, deltas)
        live = design > 0.0
        deviation = self.params.sigma / design.masked_fill(~live, 1.0).sqrt()
        innovation = ((estimate - deltas.double()) / deviation).masked_fill(~live, 0.0)
        self.memory = self.phi * self.memory + (1.0 - self.phi**2) ** 0.5 * innovation

        return (
            torch.where(live, deltas.double() + self.memory * deviation, estimate),
            design,
        )


class Bernoulli(Normal):
    """
    Binary outcomes: arm i converts at `rate + effect_i`, an epoch at allocation
    `a` gives it `round(a_i * trials)` draws and reports the success rate, at the
    realized share `n_i / trials` as the design; an arm with no draw gets no
    reading. `params.sigma` is what the policies are told and nothing here reads
    it; `describe` sets it to a success-rate reading's own noise,
    `sqrt(rate (1 - rate) / trials)`, so a spec does not write it. Each reading
    spends
    the same standard-normal variate the Gaussian world would, mapped through the
    binomial quantile function, so reps stay paired and independent of the batch.
    """

    @classmethod
    @override
    def describe(cls, fields: dict[str, object], options: dict[str, object]) -> Params:
        if "sigma" in fields:
            raise ValueError("a bernoulli world derives sigma from rate and trials")
        rate, trials = float(options.get("rate", 0.05)), int(options.get("trials", 100))

        return Params(**fields, sigma=(rate * (1.0 - rate) / trials) ** 0.5)

    def __init__(
        self,
        params: Params,
        seeds: list[int],
        device: str = "cpu",
        rate: float = 0.05,
        trials: int = 100,
    ) -> None:
        if not 0.0 < rate < 1.0 or trials < 1:
            raise ValueError(f"need 0 < rate < 1 and trials >= 1, got {rate}, {trials}")
        super().__init__(params, seeds, device)
        self.rate = rate
        self.trials = trials

    @override
    def observe(self, allocation: Tensor, deltas: Tensor) -> tuple[Tensor, Tensor]:
        from scipy import special, stats

        estimate, design = super().observe(allocation, deltas)
        live = design > 0.0
        deviation = self.params.sigma / design.masked_fill(~live, 1.0).sqrt()
        # The Gaussian world's standardized draw, as a uniform.
        z = ((estimate - deltas.double()) / deviation).masked_fill(~live, 0.0)
        uniform = special.ndtr(z.cpu().numpy())
        count = torch.round(design * self.trials).long()
        probability = (self.rate + deltas.double()).clamp(1e-9, 1.0 - 1e-9)
        successes = stats.binom.ppf(
            uniform, count.cpu().numpy(), probability.cpu().numpy()
        )
        successes = torch.as_tensor(successes, dtype=torch.float64, device=self.device)
        drawn = count > 0
        share = count.double() / self.trials
        rate = successes / count.double().masked_fill(~drawn, 1.0)

        return torch.where(drawn, rate - self.rate, 0.0), share


def matern52(distance: Tensor, lengthscale: float) -> Tensor:
    """The Matern-5/2 correlation at `distance`, unit amplitude."""
    r = 5.0**0.5 * distance / lengthscale

    return (1.0 + r + r**2 / 3.0) * torch.exp(-r)


class Matern(Normal):
    """
    A bid ladder: arm i sits at bid i, and the arms' effects are one draw from a
    Gaussian process with mean `effect`, amplitude `effect_std` and a Matern-5/2
    kernel at `lengthscale` (in bid steps), so neighbours' profits are alike and
    the smoothness is the correlation. No control: every arm is a draw. Drift
    and observation noise are `Normal`'s.
    """

    def __init__(
        self,
        params: Params,
        seeds: list[int],
        device: str = "cpu",
        lengthscale: float = 1.0,
    ) -> None:
        if lengthscale <= 0.0:
            raise ValueError(f"lengthscale must be positive, got {lengthscale}")
        self.lengthscale = lengthscale
        super().__init__(params, seeds, device)
        bids = torch.arange(params.arms, dtype=torch.float64)
        kernel = matern52((bids[:, None] - bids[None, :]).abs(), lengthscale)
        self.covariance = (params.effect_std**2 * kernel).to(device)
        self.root = torch.linalg.cholesky(
            self.covariance
            + 1e-12 * torch.eye(params.arms, dtype=torch.float64, device=device)
        )

    @property
    @override
    def effect_draws(self) -> int:
        return self.params.arms

    @override
    def draw_effect(self) -> Tensor:
        z = torch.stack([self.normal(0.0, 1.0) for _ in range(self.params.arms)], dim=1)

        return (self.params.effect + z @ self.root.T).float()

    @override
    def prior(self) -> tuple[Tensor, Tensor]:
        mean = torch.full(
            (self.params.arms,),
            self.params.effect,
            dtype=torch.float64,
            device=self.device,
        )

        return mean, self.covariance.clone()


ENVIRONMENTS: dict[str, type[Environment]] = {
    "normal": Normal,
    "student": Student,
    "ar1": AR1,
    "bernoulli": Bernoulli,
    "matern": Matern,
}
"""The worlds a spec may name, by `kind`."""


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

    ETA_FACTOR: ClassVar[float] = 1.0
    """What the policy believes `eta` is, as a multiple of the truth; 0 is a
    filter that does not know the world moves."""

    PRIOR: ClassVar[str] = "flat"
    """Where the posterior starts: `flat` is the policy's own default, `world`
    the environment's effect law (`prime`), as much of it as the filter holds."""

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
        return cls(
            params,
            reps,
            device,
            sigma=cls.SIGMA_FACTOR * params.sigma,
            eta=cls.ETA_FACTOR * params.eta,
        )

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

    def prime(self, mean: Tensor, cov: Tensor) -> None:
        """
        Start from a Gaussian over the arms. This filter is independent per arm,
        so it keeps the diagonal; a known arm (zero variance) is pinned.
        """
        variance = torch.diagonal(cov)
        self.mean = mean.double().expand(self.reps, -1).clone()
        self.precision = (
            (1.0 / variance.clamp_min(1e-12)).double().expand(self.reps, -1).clone()
        )

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
    from .policies import ALL, ProbabilityMatching

    params = Params(
        gamma=0.999,
        horizon=300,
        sigma=1.0,
        effect=0.3,
        effect_std=0.4,
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

    # Student: unit-variance t noise, and the same paired-seed invariance.
    params = replace(params, arms=3)
    fat = Student(params, list(range(reps)), df=3.0)
    fat.draw_effect()
    zeros = torch.zeros(reps, 3)
    draws = torch.stack(
        [fat.observe(torch.full((reps, 3), 1.0), zeros)[0] for _ in range(300)]
    )
    assert abs(float(draws.var()) - 1.0) < 0.15, float(draws.var())
    assert float(draws.abs().max()) > 4.0
    batch = Student(params, list(range(reps)), df=3.0).run(
        ProbabilityMatching.init(params, reps, "cpu"),
        Student(params, list(range(reps)), df=3.0).draw_effect(),
    )
    sub = Student(params, [5, 2, 7], df=3.0).run(
        ProbabilityMatching.init(params, 3, "cpu"),
        Student(params, [5, 2, 7], df=3.0).draw_effect(),
    )
    assert torch.allclose(sub.regret, batch.regret[[5, 2, 7]], rtol=1e-2, atol=1e-6)
    print("student: unit variance, heavy tails, paired seeds")

    # AR(1): unit variance, lag-one correlation phi, the same paired-seed invariance.
    sticky = AR1(params, list(range(reps)), phi=0.7)
    sticky.draw_effect()
    draws = torch.stack(
        [sticky.observe(torch.full((reps, 3), 1.0), zeros)[0] for _ in range(400)]
    )
    lagged = (draws[1:] * draws[:-1]).mean() / draws.var()
    assert abs(float(draws.var()) - 1.0) < 0.15, float(draws.var())
    assert abs(float(lagged) - 0.7) < 0.1, float(lagged)
    batch = AR1(params, list(range(reps)), phi=0.7).run(
        ProbabilityMatching.init(params, reps, "cpu"),
        AR1(params, list(range(reps)), phi=0.7).draw_effect(),
    )
    sub = AR1(params, [5, 2, 7], phi=0.7).run(
        ProbabilityMatching.init(params, 3, "cpu"),
        AR1(params, [5, 2, 7], phi=0.7).draw_effect(),
    )
    assert torch.allclose(sub.regret, batch.regret[[5, 2, 7]], rtol=1e-2, atol=1e-6)

    # A filter told eta = 0 in a moving world keeps a precision the aware one sheds.
    moving = replace(params, eta=0.05)
    unaware = type("Unaware", (ProbabilityMatching,), {"ETA_FACTOR": 0.0}).init(
        moving, reps, "cpu"
    )
    aware = ProbabilityMatching.init(moving, reps, "cpu")
    assert unaware.eta == 0.0 and aware.eta == 0.05
    print(
        "ar1: unit variance, lag-one correlation, paired seeds; eta_factor 0 is unaware"
    )

    # Bernoulli: the reading is a success rate minus the base rate, with the
    # variance of the binomial at that many draws, and reps stay paired.
    binomial_sigma = (0.05 * 0.95 / 1000) ** 0.5
    binary_params = replace(params, sigma=binomial_sigma, effect=0.0, effect_std=0.0035)
    world = Bernoulli(binary_params, list(range(reps)), rate=0.05, trials=1000)
    world.draw_effect()
    readings = torch.stack(
        [world.observe(torch.full((reps, 3), 1 / 3), zeros)[0] for _ in range(300)]
    )
    expected = (0.05 * 0.95 / (1000 / 3)) ** 0.5
    assert abs(float(readings.mean())) < 0.002, float(readings.mean())
    assert abs(float(readings.std()) / expected - 1.0) < 0.15, float(readings.std())
    _, share = world.observe(torch.tensor([[1.0, 0.0, 0.0]] * reps), zeros)
    assert torch.equal(share[0], torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    batch = Bernoulli(binary_params, list(range(reps)), rate=0.05, trials=30).run(
        ProbabilityMatching.init(binary_params, reps, "cpu"),
        Bernoulli(binary_params, list(range(reps)), rate=0.05, trials=30).draw_effect(),
    )
    sub = Bernoulli(binary_params, [5, 2, 7], rate=0.05, trials=30).run(
        ProbabilityMatching.init(binary_params, 3, "cpu"),
        Bernoulli(binary_params, [5, 2, 7], rate=0.05, trials=30).draw_effect(),
    )
    assert torch.allclose(sub.regret, batch.regret[[5, 2, 7]], rtol=1e-2, atol=1e-9)
    print("bernoulli: binomial readings at their own sigma, paired seeds")

    # Matern: the prior is the draw law, neighbours correlate as the kernel says,
    # a primed filter starts from the diagonal, and reps stay paired.
    ladder = Matern(params, list(range(2000)), lengthscale=2.0)
    effects = ladder.draw_effect().double()
    mean, cov = ladder.prior()
    assert abs(float(effects.mean()) - params.effect) < 0.05
    sample = torch.cov(effects.T)
    assert (sample - cov).abs().max() < 0.03, float((sample - cov).abs().max())
    assert (
        abs(
            float(cov[0, 1] / cov[0, 0])
            - float(matern52(torch.tensor(1.0, dtype=torch.float64), 2.0))
        )
        < 1e-12
    )
    assert abs(float(matern52(torch.tensor(1.0), 1.0)) - 0.5240) < 1e-3
    primed = ProbabilityMatching.init(params, 3, "cpu")
    primed.prime(mean, cov)
    assert torch.allclose(primed.mean[0], mean) and torch.allclose(
        primed.precision[0], 1.0 / torch.diagonal(cov)
    )
    batch = Matern(params, list(range(reps)), lengthscale=2.0).run(
        ProbabilityMatching.init(params, reps, "cpu"),
        Matern(params, list(range(reps)), lengthscale=2.0).draw_effect(),
    )
    sub = Matern(params, [5, 2, 7], lengthscale=2.0).run(
        ProbabilityMatching.init(params, 3, "cpu"),
        Matern(params, [5, 2, 7], lengthscale=2.0).draw_effect(),
    )
    assert torch.allclose(sub.regret, batch.regret[[5, 2, 7]], rtol=1e-2, atol=1e-6)
    print("matern: kernel prior, primed filter, paired seeds")


if __name__ == "__main__":
    demo()
