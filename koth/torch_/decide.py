"""The subset selection on torch."""

from __future__ import annotations

import torch

from functools import cache
from itertools import combinations
from torch import Tensor
from typing import override

from .. import decide
from ..net import Net
from .state import State
from .three_arm import ThreeArm
from .two_arm import TwoArm


@cache
def _subsets(arms: int, k: int, device: torch.device) -> Tensor:
    """Every k-subset of the arms, `(K, k)`, the first member the subset's control."""
    return torch.tensor(list(combinations(range(arms), k)), device=device)


def _pick(x: Tensor, best: Tensor) -> Tensor:
    """Row `best` of the subset axis of `x`, `(..., K, *rest)` to `(..., *rest)`."""
    dim = best.ndim
    index = best.reshape(best.shape + (1,) * (x.ndim - dim))
    index = index.expand(*best.shape, 1, *x.shape[dim + 1 :])

    return x.gather(dim, index).squeeze(dim)


class Decider(decide.Decider[Tensor]):
    """Top-k on the torch nets."""

    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        self.nets: dict[int, Net[Tensor]] = {
            net.K: net for net in (TwoArm(dtype), ThreeArm(dtype))
        }

    @override
    def decide(self, state: State, k: int) -> decide.Decision[Tensor]:
        net = self.nets[k]
        subsets = _subsets(state.arms, k, state.mean.device)
        mean = state.mean[..., subsets]
        cov = state.cov[..., subsets[:, :, None], subsets[:, None, :]]
        contrast = mean[..., 1:] - mean[..., :1]
        precision = torch.linalg.inv(
            cov[..., 1:, 1:] - cov[..., 1:, :1] - cov[..., :1, 1:] + cov[..., :1, :1]
        )
        value = net.subset_value(contrast, precision) + mean[..., 0]
        best = value.argmax(-1)
        roles = net.subset_policy(_pick(contrast, best), _pick(precision, best))
        contenders = subsets[best]
        allocation = torch.zeros(
            state.mean.shape, dtype=roles.dtype, device=roles.device
        )
        allocation.scatter_(-1, contenders, roles)
        top, arm = allocation.max(-1)
        committed = torch.where(top >= 1.0, arm, -1)

        return decide.Decision(
            allocation=allocation,
            committed=committed,
            contenders=contenders,
            value=_pick(value, best),
        )


if __name__ == "__main__":
    import numpy as np

    from ..numpy_.decide import Decider as NumpyDecider
    from ..numpy_.state import State as NumpyState

    rng = np.random.default_rng(1)
    mean = rng.normal(size=(200, 6))
    root = rng.normal(size=(200, 6, 6))
    cov = root @ root.transpose(0, 2, 1) + 0.1 * np.eye(6)
    numpy_decider, decider = NumpyDecider(), Decider()

    for k in (2, 3):
        theirs = numpy_decider.decide(NumpyState(mean, cov), k)
        mine = decider.decide(State(torch.as_tensor(mean), torch.as_tensor(cov)), k)
        assert torch.equal(mine.contenders, torch.as_tensor(theirs.contenders))
        assert torch.equal(mine.committed, torch.as_tensor(theirs.committed))
        assert np.allclose(mine.value.numpy(), theirs.value, rtol=1e-8)
        gap = np.abs(mine.allocation.numpy() - theirs.allocation).max()
        assert gap < 1e-3, (k, gap)
    print("decide on torch: matches numpy at k = 2 and 3 on 200 six-arm states")
