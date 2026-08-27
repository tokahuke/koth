"""The belief on numpy."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from typing import Self, override

from .. import state


@dataclass(frozen=True)
class State(state.State[np.ndarray]):
    """A joint Gaussian belief over the arm effects, on numpy."""

    @classmethod
    @override
    def flat(cls, arms: int, std: float) -> Self:
        return cls(np.zeros(arms), std**2 * np.eye(arms))

    @classmethod
    @override
    def independent(cls, mean: np.ndarray, var: np.ndarray) -> Self:
        mean, var = np.broadcast_arrays(np.asarray(mean, float), np.asarray(var, float))

        return cls(mean, var[..., None] * np.eye(mean.shape[-1]))

    @override
    def observe(self, estimate: np.ndarray, precision: np.ndarray) -> Self:
        estimate = np.broadcast_to(np.asarray(estimate, float), self.mean.shape)
        precision = np.broadcast_to(np.asarray(precision, float), self.mean.shape)
        information = np.linalg.inv(self.cov) + precision[..., None] * np.eye(self.arms)
        cov = np.linalg.inv(information)
        mean = (
            cov
            @ (
                np.linalg.solve(self.cov, self.mean[..., None])[..., 0]
                + precision * estimate
            )[..., None]
        )[..., 0]

        return type(self)(mean, cov)


if __name__ == "__main__":
    flat = State.flat(3, 10.0)
    assert flat.arms == 3 and np.array_equal(flat.cov, 100.0 * np.eye(3))

    # Independent arms: observing everything at huge precision lands on the estimate.
    prior = State.independent([0.0, 1.0, 2.0], [4.0, 4.0, 4.0])
    sharp = prior.observe([5.0, 6.0, 7.0], 1e9)
    assert np.allclose(sharp.mean, [5.0, 6.0, 7.0]) and np.allclose(
        sharp.cov, 0.0, atol=1e-8
    )

    # One arm, precision q: the textbook scalar update, the others untouched.
    once = prior.observe([5.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert np.isclose(once.mean[0], (0.0 / 4.0 + 5.0) / (1.0 / 4.0 + 1.0))
    assert np.isclose(once.cov[0, 0], 1.0 / (1.0 / 4.0 + 1.0))
    assert np.allclose(once.mean[1:], prior.mean[1:]) and np.allclose(
        once.cov[1:, 1:], prior.cov[1:, 1:]
    )

    # A correlated prior: observing arm 0 exactly conditions arm 1 by Gaussian
    # conditioning: mean shift cov01 / cov00 * (x - m0), variance
    # cov11 - cov01**2 / cov00.
    cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    joint = State(np.array([0.0, 1.0]), cov)
    told = joint.observe([2.0, 0.0], [1e12, 0.0])
    assert np.isclose(told.mean[1], 1.0 + 1.0 / 4.0 * 2.0)
    assert np.isclose(told.cov[1, 1], 2.0 - 1.0 / 4.0)

    # Batched over a leading axis.
    batch = State.independent(np.zeros((5, 3)), np.ones((5, 3))).observe(
        np.ones((5, 3)), 1.0
    )
    assert batch.mean.shape == (5, 3) and np.allclose(batch.mean, 0.5)
    print("state on numpy: ok")
