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
    # Test.observe: each reading weighs allocation / sigma**2, so a run of epochs
    # is the precision-weighted mean at the summed weight; all at once equals one
    # epoch at a time, and an arm at allocation 0 is untouched.
    test = Test(rho=0.01, sigma=2.0)
    outcomes = rng.normal(size=(6, 3))
    allocations = rng.dirichlet(np.ones(3), size=6)
    allocations[:, 2] = 0.0
    prior = State.flat(3, 100.0)
    at_once = test.observe(prior, outcomes, allocations)
    weight = allocations.sum(0)
    expected = State.flat(3, 100.0).update(
        (allocations * outcomes).sum(0) / np.where(weight > 0, weight, 1.0),
        weight / 4.0,
    )
    assert np.allclose(at_once.mean, expected.mean) and np.allclose(
        at_once.cov, expected.cov
    )
    assert at_once.cov[2, 2] == prior.cov[2, 2] and at_once.mean[2] == 0.0
    stepwise = prior

    for t in range(6):
        stepwise = test.observe(stepwise, outcomes[t : t + 1], allocations[t : t + 1])
    assert np.allclose(stepwise.mean, at_once.mean) and np.allclose(
        stepwise.cov, at_once.cov
    )
    print("koth: units, validation and observe ok, numpy only")
