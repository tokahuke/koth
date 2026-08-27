"""
The arena's CLI: sweep every strategy that plays the arm count against a drawn
effect, pickle the Study. Mounted by poetry as `koth-arena` (pyproject
[project.scripts]): `poetry run koth-arena simulate ...`.
"""

from __future__ import annotations

import click
import pickle
import sys
import time
import torch

from collections.abc import Callable
from pathlib import Path

from .spec import ArenaSpec
from . import policies
from .harness import Normal, Params, Policy, Run, Study, UnsupportedNumberOfArms

CHUNK = 4096
"""Reps per batched chunk, which is what caps the noise buffer."""


@click.group()
def cli() -> None:
    """Simulate a policy sweep, then report it."""


@cli.command()
@click.argument("runs", type=click.Path(dir_okay=False, path_type=Path), required=False)
@click.option(
    "--spec",
    "spec_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="A YAML arena spec (koth/arena/spec.py); authoritative, so the runs "
    "argument and the parameter flags must stay unset with it.",
)
@click.option("--gamma", type=float, default=None)
@click.option("--horizon", type=int, default=None)
@click.option("--sigma", type=float, default=None)
@click.option("--effect", type=float, default=None)
@click.option("--effect-std", type=float, default=0.0, show_default=True)
@click.option(
    "--eta",
    type=float,
    default=0.0,
    show_default=True,
    help="Drift volatility of every arm's effect per epoch; 0 is a static world.",
)
@click.option("--size", type=int, default=None)
@click.option("--arms", type=int, default=None, help="Arm count, control included.")
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Torch CPU threads; default all cores. Fewer keeps the laptop usable.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Torch device for the batched simulation.",
)
def simulate(
    runs: Path | None,
    spec_path: Path | None,
    gamma: float | None,
    horizon: int | None,
    sigma: float | None,
    effect: float | None,
    effect_std: float,
    eta: float,
    size: int | None,
    arms: int | None,
    workers: int | None,
    device: str,
) -> None:
    """
    Sweep every strategy that plays `--arms` against one drawn environment, and pickle
    the Study for `analyze`.
    """
    if spec_path is not None:
        given = (runs, gamma, horizon, sigma, effect, size, arms)

        if any(value is not None for value in given):
            raise click.UsageError("--spec is authoritative; drop RUNS and the flags")

        spec = ArenaSpec.load(spec_path)
    else:
        needed = (runs, gamma, horizon, sigma, effect, size, arms)

        if any(value is None for value in needed):
            raise click.UsageError(
                "RUNS, --gamma, --horizon, --sigma, --effect, --size and --arms are "
                "required, or --spec"
            )

        spec = ArenaSpec(
            out_path=runs,
            params=Params(
                gamma=gamma,
                horizon=horizon,
                sigma=sigma,
                effect=effect,
                effect_std=effect_std,
                size=size,
                eta=eta,
                arms=arms,
            ),
            workers=workers,
            device=device,
        )

    if spec.workers is not None:
        torch.set_num_threads(spec.workers)

    # The sweep's record: the resolved spec beside the pickle, compose-style.
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)
    spec.save(spec.out_path.with_suffix(".spec.yaml"))
    params = spec.params
    size, device = spec.params.size, spec.device
    classes = list(policies.ALL)

    # Contestants are the whole roster: each label runs the strategy it names,
    # and nothing unnamed runs. An empty list keeps every strategy.
    if len(spec.contestants) > 0:
        by_name = {cls.__name__: cls for cls in classes}
        roster: list[type[Policy]] = []

        for label, entry in spec.contestants.items():
            if entry not in by_name:
                raise click.UsageError(
                    f"contestant {label!r}: {entry!r} is not a strategy ({sorted(by_name)})"
                )

            roster.append(type(label, (by_name[entry],), {}))
        classes = roster
    results: list[Run] = []
    total = size * len(classes)

    def save() -> None:
        """Pickle every run finished so far."""
        spec.out_path.write_bytes(pickle.dumps(Study(params=params, runs=results)))

    started = time.monotonic()

    for cls in classes:
        for chunk_start in range(0, size, CHUNK):
            seeds = list(range(chunk_start, min(chunk_start + CHUNK, size)))
            # Seeded by *rep*, so comparisons are paired and a rep's stream is
            # independent of its chunk.
            world = Normal(params, seeds, device)

            try:
                policy = cls.init(params, len(seeds), device)
            except UnsupportedNumberOfArms as refused:
                print(f"skipped {cls.__name__}: {refused}", file=sys.stderr)
                total -= size

                break

            def progress(epoch: int) -> None:
                """One status line per percent of the horizon."""
                if (
                    epoch % max(1, params.horizon // 100) == 0
                    or epoch == params.horizon
                ):
                    elapsed = time.monotonic() - started
                    print(
                        f"\r{cls.__name__:<22} {seeds[0]}-{seeds[-1]}"
                        f"  epoch {epoch}/{params.horizon}  {len(results)}/{total} runs"
                        f"  {elapsed / 60:.1f} min",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )

            # Ctrl+C keeps every policy that finished: the pickle is written with
            # them and the sweep stops, so a long run can be cut and still read.
            try:
                batch = world.run(policy, world.draw_effect(), progress)
            except KeyboardInterrupt:
                print(
                    f"\ninterrupted in {cls.__name__}; saving what finished",
                    file=sys.stderr,
                )
                save()

                raise SystemExit(130)

            results.extend(batch.runs())
            print(file=sys.stderr)

    save()


@cli.command()
@click.argument("runs", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def analyze(runs: Path) -> None:
    """
    The report table: per policy, mean regret with 95% CI, ratio vs the best,
    wrong-commit share, commit share, and the median commit epoch. Under drift read
    regret, not `wrong%`.
    """
    study: Study = pickle.loads(runs.read_bytes())
    by_policy: dict[str, list[Run]] = {}

    for run in study.runs:
        by_policy.setdefault(run.policy, []).append(run)

    def mean_ci(values: list[float]) -> tuple[float, float]:
        """The mean and its 95% half-width, normal approximation."""
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)

        return mean, 1.96 * (variance / len(values)) ** 0.5

    best = min(mean_ci([r.regret for r in runs_])[0] for runs_ in by_policy.values())

    print(study.params)
    print(
        f"{'policy':<22} {'regret':>9} {'95% CI':>8} {'vs best':>8}"
        f" {'wrong%':>7} {'commit%':>8} {'median epoch':>13} {'precision time':>15}"
        f" {'off best':>12}"
    )

    for name, runs_ in sorted(
        by_policy.items(), key=lambda item: mean_ci([r.regret for r in item[1]])[0]
    ):
        mean, ci = mean_ci([r.regret for r in runs_])
        committed = [r for r in runs_ if r.committed is not None]
        wrong = [r for r in committed if r.delta[r.committed] < max(r.delta)]
        # committed_at, not epochs: the runner plays the full horizon, so epochs says
        # nothing about commitment. Studies predating the field read as None and drop
        # out of the median, like precision_time below.
        epochs = sorted(r.committed_at for r in committed if r.committed_at is not None)
        median = epochs[len(epochs) // 2] if len(epochs) > 0 else None
        # Old studies predate the field; they read as 0.
        info, info_ci = mean_ci([getattr(r, "precision_time", 0.0) for r in runs_])
        off, off_ci = mean_ci([getattr(r, "off_best", 0.0) for r in runs_])
        print(
            f"{name:<22} {mean:>9.1f} {ci:>8.1f} {mean / best:>8.2f}"
            f" {100 * len(wrong) / len(runs_):>6.1f}%"
            f" {100 * len(committed) / len(runs_):>7.1f}%"
            f" {median if median is not None else 'never':>13}"
            f" {info:>9.1f} +/-{info_ci:<4.1f}"
            f" {off:>6.1f} +/-{off_ci:<4.1f}"
        )

    _paired(by_policy, mean_ci)


def _paired(
    by_policy: dict[str, list[Run]],
    mean_ci: Callable[[list[float]], tuple[float, float]],
) -> None:
    """
    The same comparison, paired by rep, which is the one to read: the per-rep
    difference cancels the environment, and that is nearly all of the variance. Also
    prints the reps needed for a 2-sigma read on a 2% effect, which is how the *next*
    sweep should be sized.
    """
    ranked = sorted(
        by_policy.items(), key=lambda item: mean_ci([r.regret for r in item[1]])[0]
    )
    best_name, best_runs = ranked[0]

    print(f"\npaired against {best_name}, per rep (same effects, same noise)")
    print(
        f"{'policy':<22} {'difference':>12} {'95% CI':>9} {'unpaired CI':>12} {'reps for 2%':>12}"
    )

    for name, runs_ in ranked[1:]:
        # Identical draws are the whole premise; if the reps do not line up,
        # say so rather than quietly differencing unrelated runs.
        if len(runs_) != len(best_runs) or any(
            a.delta != b.delta for a, b in zip(runs_, best_runs)
        ):
            print(f"{name:<22} {'reps do not align, not paired':>50}")

            continue

        gaps = [a.regret - b.regret for a, b in zip(runs_, best_runs)]
        mean, ci = mean_ci(gaps)
        _, loose = mean_ci([r.regret for r in runs_])
        deviation = ci / 1.96 * len(gaps) ** 0.5
        target = 0.02 * sum(r.regret for r in best_runs) / len(best_runs)
        needed = (2.0 * deviation / target) ** 2

        print(f"{name:<22} {mean:>12.1f} {ci:>9.1f} {loose:>12.1f} {needed:>12.0f}")


ENTITIES: dict[str, tuple[str, str]] = {
    "Koth3": ("koth, k = 3", "#2a78d6"),
    "Koth2": ("koth, k = 2", "#7aa8e0"),
    "ProbabilityMatching": ("Thompson sampling", "#eb6834"),
    "Gittins": ("Gittins index", "#1baf7a"),
    "Elimination": ("z-test at 5%", "#eda100"),
    "ExploreThenCommit": ("explore-then-commit", "#e87ba4"),
}
"""Label and colour per strategy: colour follows the entity, never its rank."""

INK, MUTED = "#1a1f26", "#5b6472"
"""Text tokens: values and labels wear ink, never the series colour."""


@cli.command()
@click.argument("runs", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the figure; default beside the pickle, as .png.",
)
@click.option(
    "--drop",
    multiple=True,
    help="A strategy to leave out of the figure (repeatable).",
)
def plot(runs: Path, out: Path | None, drop: tuple[str, ...]) -> None:
    """
    The report as a figure: per policy, discounted regret and epochs spent
    exploring (the soft commit time), mean with its 95% CI, ordered by regret.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    study: Study = pickle.loads(runs.read_bytes())
    by_policy: dict[str, list[Run]] = {}

    for run in study.runs:
        if run.policy not in drop:
            by_policy.setdefault(run.policy, []).append(run)

    def mean_ci(values: list[float]) -> tuple[float, float]:
        """The mean and its 95% half-width, normal approximation."""
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)

        return mean, 1.96 * (variance / len(values)) ** 0.5

    ranked = sorted(
        by_policy, key=lambda name: mean_ci([r.regret for r in by_policy[name]])[0]
    )
    labels = [ENTITIES.get(name, (name, "#8a93a1"))[0] for name in ranked]
    colors = [ENTITIES.get(name, (name, "#8a93a1"))[1] for name in ranked]
    panels = {
        "discounted regret  (lower is better)": [
            mean_ci([r.regret for r in by_policy[n]]) for n in ranked
        ],
        "discounted traffic sent to losing arms  (lower is better)": [
            mean_ci([r.off_best for r in by_policy[n]]) for n in ranked
        ],
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(ranked) + 1.6))
    y = list(range(len(ranked)))[::-1]

    for ax, (title, values) in zip(axes, panels.items()):
        means = [m for m, _ in values]
        cis = [ci for _, ci in values]
        ax.barh(
            y,
            means,
            xerr=cis,
            color=colors,
            height=0.62,
            error_kw={"ecolor": INK, "capsize": 3, "elinewidth": 1},
        )
        span = max(m + ci for m, ci in values)

        for position, mean, ci in zip(y, means, cis):
            ax.text(
                mean + ci + 0.015 * span,
                position,
                f"{mean:.1f}",
                va="center",
                fontsize=10,
                color=INK,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, color=INK)
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlim(0, 1.12 * span)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.xaxis.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_axisbelow(True)
    reps = len(by_policy[ranked[0]])
    figure.suptitle(
        f"{study.params.arms} independent arms, {reps} random tests each",
        fontsize=16,
        color=INK,
        x=0.02,
        ha="left",
    )
    figure.text(
        0.02,
        -0.01,
        f"Regret: profit left on the table vs an oracle that picks the winner from "
        f"epoch one, discounted at gamma = {study.params.gamma} over "
        f"{study.params.horizon} epochs.\nLosing arms: the share of each epoch's "
        "traffic not on the true best arm, discounted the same way; a whole epoch "
        "on a loser counts one.",
        fontsize=8,
        color="#8a93a1",
        va="top",
        linespacing=1.45,
    )
    figure.tight_layout(rect=[0, 0.02, 1, 0.94])
    out = runs.with_suffix(".png") if out is None else out
    figure.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    cli()
