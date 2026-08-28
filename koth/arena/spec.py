"""
Arena specs: one YAML file describing one sweep, the way a compose file
describes a service. `koth-arena simulate --spec` runs one, and every sweep
writes its resolved spec beside the pickle it produces.
"""

from __future__ import annotations

import yaml

from dataclasses import MISSING, dataclass, field, fields, replace
from pathlib import Path
from typing import Self

from .harness import ENVIRONMENTS, Params

_KEYS = {"out", "size", "environments", "workers", "device", "contestants"}
"""Every top-level key a spec file may carry; anything else raises."""

_FIELDS = {field_.name for field_ in fields(Params)}
"""The `Params` fields, which every world spells out (anchors and `<<` merges
keep the repetition in the file down)."""


@dataclass
class ArenaSpec:
    """One arena sweep, fully specified."""

    out_path: Path
    """The Study pickle to write (`out` in the file), for `koth-arena analyze`."""

    size: int
    """Tests per world; seeds are the test indices, so worlds and contestants pair."""

    environments: dict[str, dict[str, object]]
    """
    The worlds, report label -> mapping: the `Params` fields, spelled out for
    every world, a `kind` naming an entry of `harness.ENVIRONMENTS` (`normal`
    when absent), an optional `told` mapping of `Params` fields that every
    strategy is told instead of the truth (`told: {sigma: 1.36}`), and anything
    else passed to that environment's constructor. Every contestant plays every
    world; runs carry the world's label and the label is the figure's axis text.
    """

    workers: int | None = None
    """Torch CPU threads; None keeps torch's default (all cores)."""

    device: str = "cpu"
    """Torch device for the batched simulation."""

    contestants: dict[str, dict[str, object]] = field(default_factory=dict)
    """
    The complete roster when nonempty, report label -> `{strategy: Name, ...}`,
    every other key a class attribute set on that strategy for this contestant,
    uppercased (`sigma_factor: 0.5` sets `SIGMA_FACTOR`). A bare name is
    `{strategy: Name}`. Nothing unnamed runs. Empty sweeps every strategy.
    """

    def worlds(self) -> dict[str, tuple[Params, Params, str, dict[str, object]]]:
        """
        Each world as `(truth, told, kind, constructor options)`, by label: the
        environment runs on `truth`, the strategies are built on `told`.
        """
        out: dict[str, tuple[Params, Params, str, dict[str, object]]] = {}

        for name, entry in self.environments.items():
            kind = str(entry.get("kind", "normal"))

            if kind not in ENVIRONMENTS:
                raise ValueError(
                    f"world {name!r}: {kind!r} is not one of {sorted(ENVIRONMENTS)}"
                )
            fields = {k: v for k, v in entry.items() if k in _FIELDS}
            options = {
                k: v
                for k, v in entry.items()
                if k not in _FIELDS and k not in ("kind", "told")
            }
            told = {str(k): v for k, v in (entry.get("told") or {}).items()}

            if len(set(told) - _FIELDS) > 0:
                raise ValueError(
                    f"world {name!r}: told {sorted(set(told) - _FIELDS)} are not Params"
                )
            truth = ENVIRONMENTS[kind].describe(fields, options)
            out[name] = (truth, replace(truth, **told), kind, options)

        return out

    @classmethod
    def load(cls, path: Path) -> Self:
        """The spec in a YAML file, unknown keys refused."""
        raw = yaml.safe_load(path.read_text())
        unknown = set(raw) - _KEYS

        if len(unknown) > 0:
            raise ValueError(f"unknown arena spec keys: {sorted(unknown)}")
        spec = cls(
            out_path=Path(raw["out"]),
            size=int(raw["size"]),
            environments={
                str(name): {str(key): value for key, value in (entry or {}).items()}
                for name, entry in raw["environments"].items()
            },
            workers=int(raw["workers"]) if "workers" in raw else None,
            device=str(raw.get("device", "cpu")),
            contestants={
                str(name): (
                    {"strategy": str(entry)}
                    if isinstance(entry, str)
                    else {str(key): value for key, value in entry.items()}
                )
                for name, entry in (raw.get("contestants") or {}).items()
            },
        )
        spec.worlds()

        return spec

    def save(self, path: Path) -> None:
        """
        The spec as YAML, in the file's own key names, and deviations only:
        a field sitting at its default stays out of the file. Worlds are
        written out in full (an anchor in the source does not survive).
        """
        raw: dict[str, object] = {
            "out": str(self.out_path),
            "size": self.size,
            "environments": {
                name: dict(entry) for name, entry in self.environments.items()
            },
        }

        if self.workers is not None:
            raw["workers"] = self.workers

        if self.device != "cpu":
            raw["device"] = self.device

        if len(self.contestants) > 0:
            raw["contestants"] = dict(self.contestants)
        path.write_text(yaml.safe_dump(raw, sort_keys=False))


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "sweep.yaml").write_text(
            "out: probe.pkl\n"
            "size: 64\n"
            "environments:\n"
            "  normal: &base\n"
            "    gamma: 0.999\n"
            "    horizon: 100\n"
            "    sigma: 1.0\n"
            "    effect: 0.0\n"
            "    effect_std: 0.5\n"
            "    arms: 2\n"
            "  df 3: {<<: *base, kind: student, df: 3}\n"
            "  drifting: {<<: *base, eta: 0.01}\n"
            "workers: 4\n"
            "contestants:\n"
            "  ts: ProbabilityMatching\n"
            "  koth_half: {strategy: KotH3, sigma_factor: 0.5}\n"
        )
        sweep = ArenaSpec.load(root / "sweep.yaml")
        worlds = sweep.worlds()

        # Anchors and merges resolve into three full worlds.
        assert list(worlds) == ["normal", "df 3", "drifting"]
        plain = Params(
            gamma=0.999, horizon=100, sigma=1.0, effect=0.0, effect_std=0.5, arms=2
        )
        assert worlds["normal"] == (plain, plain, "normal", {})
        assert worlds["df 3"][2:] == ("student", {"df": 3})
        assert worlds["df 3"][0].eta == 0.0
        assert worlds["drifting"][0].eta == 0.01 and worlds["drifting"][2] == "normal"

        # `told` moves what the strategies believe, not the world.
        (root / "told.yaml").write_text(
            (root / "sweep.yaml")
            .read_text()
            .replace(
                "  drifting: {<<: *base, eta: 0.01}\n",
                "  lied to: {<<: *base, told: {sigma: 2.0}}\n",
            )
        )
        truth, told, _, _ = ArenaSpec.load(root / "told.yaml").worlds()["lied to"]
        assert truth.sigma == 1.0 and told.sigma == 2.0 and told.gamma == truth.gamma

        # A bernoulli world derives sigma, and refuses one written by hand.
        (root / "binary.yaml").write_text(
            (root / "sweep.yaml")
            .read_text()
            .replace(
                "  df 3: {<<: *base, kind: student, df: 3}\n",
                "  coarse: {gamma: 0.999, horizon: 100, effect: 0.0, effect_std: 0.0035, "
                "arms: 2, kind: bernoulli, rate: 0.05, trials: 100}\n",
            )
        )
        coarse = ArenaSpec.load(root / "binary.yaml").worlds()["coarse"]

        assert abs(coarse[0].sigma - (0.05 * 0.95 / 100) ** 0.5) < 1e-12
        assert coarse[1].sigma == coarse[0].sigma
        (root / "binary_bad.yaml").write_text(
            (root / "sweep.yaml")
            .read_text()
            .replace(
                "  df 3: {<<: *base, kind: student, df: 3}\n",
                "  coarse: {<<: *base, kind: bernoulli, rate: 0.05, trials: 100}\n",
            )
        )

        try:
            ArenaSpec.load(root / "binary_bad.yaml")
            raise AssertionError("a hand-written sigma on a bernoulli world must raise")
        except ValueError as e:
            assert "derives sigma" in str(e)
        assert sweep.size == 64 and sweep.workers == 4
        assert sweep.contestants["ts"] == {"strategy": "ProbabilityMatching"}
        assert sweep.contestants["koth_half"] == {
            "strategy": "KotH3",
            "sigma_factor": 0.5,
        }

        # Roundtrip through save, worlds written in full.
        sweep.save(root / "back.yaml")
        written = (root / "back.yaml").read_text()

        assert ArenaSpec.load(root / "back.yaml") == sweep
        assert "device" not in written and written.count("gamma") == 3

        # Refusals: an unknown top-level key, a world missing a field, an unknown one.
        (root / "bad.yaml").write_text(
            (root / "sweep.yaml").read_text().replace("workers: 4", "reps: 5")
        )

        try:
            ArenaSpec.load(root / "bad.yaml")
            raise AssertionError("unknown arena keys must raise")
        except ValueError as e:
            assert "reps" in str(e)

        (root / "short.yaml").write_text(
            "out: x.pkl\nsize: 1\nenvironments:\n"
            "  normal: {gamma: 0.9, horizon: 1, sigma: 1.0, effect: 0.0, effect_std: 0.0}\n"
        )

        try:
            ArenaSpec.load(root / "short.yaml")
            raise AssertionError("a world missing a Params field must raise")
        except TypeError as e:
            assert "arms" in str(e)
    print("ok")
