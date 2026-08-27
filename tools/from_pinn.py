"""
Export pinn's champions into koth's weight files, plus a fixture of reference
values pinn computes, which koth's self-checks compare against. Runs in the
pinn repo's venv (it imports pinn); koth itself never does.

    cd ../pinn && poetry run python ../koth/tools/from_pinn.py
"""

from __future__ import annotations

import numpy as np
import torch

from pathlib import Path

from pinn.problems.three_arm.model import DimensionlessValueFunction as ThreeArm
from pinn.problems.three_arm.model import ValueFunction as ThreeArmValue
from pinn.problems.three_arm.sample import Sample
from pinn.problems.two_arm.model import DimensionlessValueFunction as TwoArm

OUT = Path(__file__).resolve().parent.parent / "koth" / "_weights"
PINN_DATA = Path(__file__).resolve().parent.parent.parent / "pinn" / "data"


def layers(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    A premium's MLP as flat arrays: `log_scale`, `feature_scale`, per layer
    `w<i>`, `b<i>` and a `g<i>` gain on every hidden one, plus the kink branch
    (`kink_in_w`, `kink_in_b`, `kink_out_w`, `kink_out_b`) when there is one.
    """
    arrays: dict[str, np.ndarray] = {
        "log_scale": state["premium.log_scale"].numpy(),
        "feature_scale": state["premium.feature_scale"].numpy(),
    }
    layer = 0

    while f"premium.net.{2 * layer}.weight" in state:
        arrays[f"w{layer}"] = state[f"premium.net.{2 * layer}.weight"].numpy()
        arrays[f"b{layer}"] = state[f"premium.net.{2 * layer}.bias"].numpy()

        if f"premium.net.{2 * layer + 1}.gain" in state:
            arrays[f"g{layer}"] = state[f"premium.net.{2 * layer + 1}.gain"].numpy()
        layer += 1

    if "premium.kink_in.weight" in state:
        arrays["kink_in_w"] = state["premium.kink_in.weight"].numpy()
        arrays["kink_in_b"] = state["premium.kink_in.bias"].numpy()
        arrays["kink_out_w"] = state["premium.kink_out.weight"].numpy()
        arrays["kink_out_b"] = state["premium.kink_out.bias"].numpy()

    return {name: value.astype(np.float64) for name, value in arrays.items()}


def export_two_arm() -> None:
    """The two-arm champion's weights and a fixture of pinn's values on it."""
    arrays = layers(torch.load(PINN_DATA / "two_arm.pt"))
    np.savez(OUT / "two_arm.npz", **arrays)

    # The fixture: states across the trained decades and below the floor, with
    # pinn's own premium, learning number and policy at each.
    net = TwoArm.load(PINN_DATA / "two_arm.pt")
    generator = torch.Generator().manual_seed(0)
    n = 2000
    muhat = 5.0 * torch.rand(n, generator=generator)
    tauhat = 10.0 ** (7.0 * torch.rand(n, generator=generator) - 4.0)
    _, best, l_ab = net.hamiltonian(muhat, tauhat)
    np.savez(
        OUT / "two_arm.check.npz",
        muhat=muhat.numpy().astype(np.float64),
        tauhat=tauhat.numpy().astype(np.float64),
        u=net.premium(muhat, tauhat).detach().numpy().astype(np.float64),
        l_ab=l_ab.detach().numpy().astype(np.float64),
        alpha=best.x.detach().numpy().astype(np.float64),
    )
    print(f"two_arm: {len(arrays)} arrays, {n} fixture states")


def export_three_arm() -> None:
    """The three-arm champion's weights and a fixture of pinn's values on it."""
    arrays = layers(torch.load(PINN_DATA / "three_arm.pt"))
    np.savez(OUT / "three_arm.npz", **arrays)

    # The fixture: precisions from the sampler (funnel included), means of
    # either sign so the fold is exercised, at rho = sigma = 1 (the dimensionless
    # form); pinn's value and physical-label policy on the raw state, and its
    # premium, learning numbers and wedge policy on the folded one.
    torch.manual_seed(0)
    n = 2000
    net = ThreeArm.load(PINN_DATA / "three_arm.pt")
    deployed = ThreeArmValue(net, rho=1.0, sigma=1.0)
    draw = Sample.draw(n)
    det = draw.tau_bb * draw.tau_cc - draw.tau_bc**2
    m_b = 2.0 * (draw.tau_cc / det).sqrt() * torch.randn(n)
    m_c = 2.0 * (draw.tau_bb / det).sqrt() * torch.randn(n)
    raw = Sample(m_b, m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)
    folded = raw.fold()
    state = (folded.m_b, folded.m_c, folded.tau_bb, folded.tau_bc, folded.tau_cc)
    _, best, (l_ab, l_ac, l_bc) = net.hamiltonian(*state)
    np.savez(
        OUT / "three_arm.check.npz",
        m_b=m_b.numpy().astype(np.float64),
        m_c=m_c.numpy().astype(np.float64),
        tau_bb=draw.tau_bb.numpy().astype(np.float64),
        tau_bc=draw.tau_bc.numpy().astype(np.float64),
        tau_cc=draw.tau_cc.numpy().astype(np.float64),
        value=deployed(*(x for x in (m_b, m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)))
        .detach()
        .numpy()
        .astype(np.float64),
        allocation=deployed.policy(m_b, m_c, draw.tau_bb, draw.tau_bc, draw.tau_cc)
        .numpy()
        .astype(np.float64),
        u=net.premium(*state).detach().numpy().astype(np.float64),
        l_ab=l_ab.detach().numpy().astype(np.float64),
        l_ac=l_ac.detach().numpy().astype(np.float64),
        l_bc=l_bc.detach().numpy().astype(np.float64),
        x=best.x.detach().numpy().astype(np.float64),
        y=best.y.detach().numpy().astype(np.float64),
    )
    print(f"three_arm: {len(arrays)} arrays, {n} fixture states")


if __name__ == "__main__":
    export_two_arm()
    export_three_arm()
