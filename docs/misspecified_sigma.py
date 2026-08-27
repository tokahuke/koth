"""
The misspecified-noise analysis: regret against the factor by which a strategy
misjudges `sigma`, one line per strategy, from the sweep
`docs/misspecified_sigma.spec.yaml` produced.

    koth-arena simulate --spec docs/misspecified_sigma.spec.yaml
    poetry run python docs/misspecified_sigma.py
"""

from __future__ import annotations

import matplotlib
import pickle
import sys

from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from koth.arena.harness import Run, Study
from koth.arena.main import COLORS, GREY, INK, MUTED

STUDY = Path(sys.argv[1] if len(sys.argv) > 1 else "data/misspecified_sigma.pkl")
OUT = Path(__file__).with_suffix(".png")


def mean_ci(values: list[float]) -> tuple[float, float]:
    """The mean and its 95% half-width, normal approximation."""
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)

    return mean, 1.96 * (variance / len(values)) ** 0.5


study: Study = pickle.loads(STUDY.read_bytes())
by_policy: dict[str, list[Run]] = {}

for run in study.runs:
    by_policy.setdefault(run.policy, []).append(run)

# Contestants are named `<figure label> x<factor>` by the spec.
curves: dict[str, list[tuple[float, float, float]]] = {}

for label, runs in by_policy.items():
    strategy, _, factor = label.rpartition(" x")
    mean, ci = mean_ci([r.regret for r in runs])
    curves.setdefault(strategy, []).append((float(factor), mean, ci))

figure, ax = plt.subplots(figsize=(8, 4.5))

for strategy, points in curves.items():
    points.sort()
    factors = [f for f, _, _ in points]
    means = [m for _, m, _ in points]
    cis = [c for _, _, c in points]
    color = COLORS.get(strategy, GREY)
    ax.plot(factors, means, marker="o", color=color, label=strategy)
    ax.fill_between(
        factors,
        [m - c for m, c in zip(means, cis)],
        [m + c for m, c in zip(means, cis)],
        color=color,
        alpha=0.15,
        linewidth=0,
    )
ax.set_xscale("log", base=2)
ax.set_xticks([f for f, _, _ in next(iter(curves.values()))])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("believed sigma / true sigma", color=MUTED)
ax.set_ylabel("discounted regret", color=MUTED)
ax.set_ylim(bottom=0.0)
ax.set_title(
    f"{study.params.arms} independent arms, {len(next(iter(by_policy.values())))} random "
    "tests per point: regret when a strategy misjudges the noise",
    loc="left",
    fontsize=11,
    color=INK,
)
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.grid(True, alpha=0.25, linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(colors=MUTED, labelsize=9)
figure.tight_layout()
figure.savefig(OUT, dpi=150, facecolor="white")
print(f"wrote {OUT}")
