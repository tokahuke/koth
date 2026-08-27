"""The subset selection on numpy."""

from __future__ import annotations

import numpy as np

from functools import cache
from itertools import combinations
from typing import override

from .. import decide
from ..net import Net
from .state import State
from .three_arm import ThreeArm
from .two_arm import TwoArm


@cache
def _subsets(arms: int, k: int) -> np.ndarray:
    """Every k-subset of the arms, `(K, k)`, the first member the subset's control."""
    return np.array(list(combinations(range(arms), k)))


def _pick(x: np.ndarray, best: np.ndarray) -> np.ndarray:
    """Row `best` of the subset axis of `x`, `(..., K, *rest)` to `(..., *rest)`."""
    axis = best.ndim
    index = best.reshape(best.shape + (1,) * (x.ndim - axis))
    index = np.broadcast_to(index, best.shape + (1,) + x.shape[axis + 1 :])

    return np.take_along_axis(x, index, axis)[
        (..., 0) + (slice(None),) * (x.ndim - axis - 1)
    ]


class Decider(decide.Decider[np.ndarray]):
    """Top-k on the numpy nets."""

    def __init__(self) -> None:
        self.nets: dict[int, Net[np.ndarray]] = {
            net.K: net for net in (TwoArm(), ThreeArm())
        }

    @override
    def decide(self, state: State, k: int) -> decide.Decision[np.ndarray]:
        net = self.nets[k]
        subsets = _subsets(state.arms, k)
        mean = state.mean[..., subsets]
        cov = state.cov[..., subsets[:, :, None], subsets[:, None, :]]
        contrast = mean[..., 1:] - mean[..., :1]
        precision = np.linalg.inv(
            cov[..., 1:, 1:] - cov[..., 1:, :1] - cov[..., :1, 1:] + cov[..., :1, :1]
        )
        value = net.subset_value(contrast, precision) + mean[..., 0]
        best = value.argmax(-1)
        roles = net.subset_policy(_pick(contrast, best), _pick(precision, best))
        contenders = subsets[best]
        allocation = np.zeros(state.mean.shape, dtype=roles.dtype)
        np.put_along_axis(allocation, contenders, roles, -1)
        committed = np.where(allocation.max(-1) >= 1.0, allocation.argmax(-1), -1)

        return decide.Decision(
            allocation=allocation,
            committed=committed,
            contenders=contenders,
            value=_pick(value, best),
        )


if __name__ == "__main__":
    decider = Decider()
    two_arm, three_arm = decider.nets[2], decider.nets[3]
    rng = np.random.default_rng(0)

    # At k = arms the selection is the net itself: two arms against TwoArm on the
    # contrast, three against ThreeArm on the contrasts' precision.
    mean = rng.normal(size=(64, 2))
    cov = np.array([[1.0, 0.3], [0.3, 2.0]]) * np.ones((64, 1, 1))
    two = decider.decide(State(mean, cov), 2)
    share = two_arm.policy(mean[:, 1] - mean[:, 0], 1.0 / (1.0 + 2.0 - 0.6))
    assert np.allclose(two.allocation[:, 1], share)
    assert np.allclose(two.allocation.sum(-1), 1.0)
    assert (two.contenders == [0, 1]).all()

    mean = rng.normal(size=(64, 3))
    cov = np.array([[1.0, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 2.0]]) * np.ones(
        (64, 1, 1)
    )
    three = decider.decide(State(mean, cov), 3)
    spread = cov[:, 1:, 1:] - cov[:, 1:, :1] - cov[:, :1, 1:] + cov[:, :1, :1]
    precision = np.linalg.inv(spread)
    roles = three_arm.policy(
        mean[:, 1] - mean[:, 0],
        mean[:, 2] - mean[:, 0],
        precision[:, 0, 0],
        precision[:, 0, 1],
        precision[:, 1, 1],
    )
    assert np.allclose(three.allocation, roles)

    # Eight arms, one far ahead and well measured: both k commit to it.
    mean = np.array([[0.0, -0.2, 1.5, -0.1, 0.0, -0.3, 0.1, -0.4]])
    cov = 0.01 * np.eye(8)[None]
    for k in (2, 3):
        run = decider.decide(State(mean, cov), k)
        assert run.committed[0] == 2 and run.allocation[0, 2] == 1.0, (k, run)
        assert 2 in run.contenders[0]

    # Batched over leading axes; a flat prior on eight arms commits to nobody.
    wide = decider.decide(
        State(rng.normal(size=(4, 5, 8)), 100.0 * np.eye(8) * np.ones((4, 5, 1, 1))),
        3,
    )
    assert wide.allocation.shape == (4, 5, 8) and wide.contenders.shape == (4, 5, 3)
    assert (wide.committed == -1).all() and np.allclose(wide.allocation.sum(-1), 1.0)

    # Value is the winning subset's, and the subset's control commit value is in it.
    lead = decider.decide(State(np.array([[3.0, 0.0]]), np.eye(2)[None]), 2)
    assert lead.value[0] >= 3.0
    print("decide on numpy: ok")
