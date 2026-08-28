"""The belief on torch."""

from __future__ import annotations

import torch

from dataclasses import dataclass
from torch import Tensor
from typing import Self, override

from .. import state


@dataclass(frozen=True)
class State(state.State[Tensor]):
    """A joint Gaussian belief over the arm effects, on torch (float64 by default)."""

    @classmethod
    @override
    def flat(cls, arms: int, std: float, dtype: torch.dtype = torch.float64) -> Self:
        return cls(
            torch.zeros(arms, dtype=dtype), std**2 * torch.eye(arms, dtype=dtype)
        )

    @classmethod
    @override
    def independent(cls, mean: Tensor, var: Tensor) -> Self:
        mean, var = torch.broadcast_tensors(mean, var)

        return cls(mean, torch.diag_embed(var))

    @override
    def update(self, estimate: Tensor, precision: Tensor) -> Self:
        estimate = estimate.broadcast_to(self.mean.shape)
        precision = precision.broadcast_to(self.mean.shape)
        information = torch.linalg.inv(self.cov) + torch.diag_embed(precision)
        cov = torch.linalg.inv(information)
        mean = (
            cov
            @ (
                torch.linalg.solve(self.cov, self.mean[..., None])[..., 0]
                + precision * estimate
            )[..., None]
        )[..., 0]

        return type(self)(mean, cov)


if __name__ == "__main__":
    import numpy as np

    from ..arena.harness import Bayesian, Params
    from ..numpy_.state import State as NumpyState

    # Against the arena's Kalman filter, independent arms, no drift: the same
    # numbers epoch for epoch, on torch and on numpy.
    params = Params(
        gamma=0.99, horizon=1, sigma=2.0, effect=0.0, effect_std=1.0, arms=3
    )

    class Filter(Bayesian):
        """The arena filter with a stub policy, for the update alone."""

        @override
        def propose(self) -> Tensor:
            return self.uniform()

    filter_ = Filter(params, 4, "cpu", prior_precision=0.25)
    belief = State.independent(
        torch.zeros(4, 3, dtype=torch.float64),
        torch.full((4, 3), 4.0, dtype=torch.float64),
    )
    numpy_belief = NumpyState.independent(np.zeros((4, 3)), np.full((4, 3), 4.0))
    generator = torch.Generator().manual_seed(0)

    for _ in range(5):
        estimate = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        design = torch.rand(4, 3, generator=generator, dtype=torch.float64)
        design[:, 2] = 0.0
        filter_.observe((estimate, design))
        belief = belief.update(estimate, design / params.sigma**2)
        numpy_belief = numpy_belief.update(
            estimate.numpy(), (design / params.sigma**2).numpy()
        )
    assert torch.allclose(belief.mean, filter_.mean) and torch.allclose(
        torch.diagonal(belief.cov, dim1=-2, dim2=-1), 1.0 / filter_.precision
    )
    assert np.allclose(numpy_belief.mean, filter_.mean.numpy())
    assert np.allclose(numpy_belief.cov, belief.cov.numpy())
    assert State.flat(3, 10.0).cov[1, 1] == 100.0

    print("state on torch: matches the arena filter and numpy over 5 epochs")
