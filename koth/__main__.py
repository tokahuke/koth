"""The package's self-check: `Test` is the readout dictionary and nothing else."""

from __future__ import annotations

import numpy as np
import sys

from . import State, Test
from .numpy_.decide import Decider

if __name__ == "__main__":
    assert "torch" not in sys.modules
    rng = np.random.default_rng(2)
    mean = rng.normal(size=(50, 5))
    root = rng.normal(size=(50, 5, 5))
    cov = root @ root.transpose(0, 2, 1) + 0.1 * np.eye(5)
    chart = Decider().decide(State(mean, cov), 3)

    # rho = sigma = 1 is the dimensionless form.
    unit = Test(rho=1.0, sigma=1.0).decide(State(mean, cov), 3)
    assert np.array_equal(unit.allocation, chart.allocation)
    assert np.array_equal(unit.value, chart.value)

    # Units: means scale by sigma sqrt(rho), covariance by rho sigma**2, and the
    # decision is the same one with the value read back through 1 / rho. Up to
    # ties: subsets differing by a dead arm score equal, and an ulp of rounding
    # picks either, so contenders are compared where the tie did not flip.
    rho, sigma = 0.002, 37.0
    scaled = Test(rho=rho, sigma=sigma).decide(
        State(mean * sigma * rho**0.5, cov * rho * sigma**2), 3
    )
    same = (scaled.contenders == chart.contenders).all(-1)
    assert same.mean() > 0.9, same.mean()
    assert np.allclose(scaled.allocation[same], chart.allocation[same], atol=1e-6)
    assert np.array_equal(scaled.committed, chart.committed)
    assert np.allclose(scaled.value, chart.value * sigma * rho**0.5 / rho, rtol=1e-9)

    # The default k is min(3, arms); 2 or 3 only, and never more than the arms.
    assert Test(1.0, 1.0).decide(
        State(mean[:, :2], cov[:, :2, :2])
    ).contenders.shape == (50, 2)
    assert Test(1.0, 1.0).decide(State(mean, cov)).contenders.shape == (50, 3)

    for bad in (1, 4):
        try:
            Test(1.0, 1.0).decide(State(mean, cov), bad)

            raise AssertionError(bad)
        except ValueError:
            pass

    try:
        Test(1.0, 1.0).decide(State(mean[:, :2], cov[:, :2, :2]), 3)

        raise AssertionError("k = 3 on 2 arms")
    except ValueError:
        pass

    try:
        State(mean, cov[:, :4, :4])

        raise AssertionError("shape mismatch")
    except ValueError:
        pass
    print("koth: units and validation ok, numpy only")
