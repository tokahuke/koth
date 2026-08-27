"""
Arena specs: one YAML file describing one sweep, the way a compose file
describes a service. `koth-arena simulate --spec` runs one, and every sweep
writes its resolved spec beside the pickle it produces.
"""

from __future__ import annotations

import yaml

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Self

from .harness import Params

_KEYS = {"out", "params", "workers", "device", "contestants"}
"""Every top-level key a spec file may carry; anything else raises. The
`params` mapping is validated by the harness's own `Params` constructor."""


@dataclass
class ArenaSpec:
    """One arena sweep, fully specified."""

    out_path: Path
    """The Study pickle to write (`out` in the file), for `koth-arena analyze`."""

    params: Params
    """The environment, exactly the harness's own dataclass (`params` in the
    file): an unknown or missing key raises from its constructor, named."""

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

    @classmethod
    def load(cls, path: Path) -> Self:
        """The spec in a YAML file, unknown keys refused."""
        raw = yaml.safe_load(path.read_text())
        unknown = set(raw) - _KEYS

        if len(unknown) > 0:
            raise ValueError(f"unknown arena spec keys: {sorted(unknown)}")

        return cls(
            out_path=Path(raw["out"]),
            params=Params(**(raw.get("params") or {})),
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

    def save(self, path: Path) -> None:
        """
        The spec as YAML, in the file's own key names, and deviations only:
        a field sitting at its default stays out of the file.
        """
        environment = {
            spec_field.name: getattr(self.params, spec_field.name)
            for spec_field in fields(self.params)
            if spec_field.default is MISSING
            or getattr(self.params, spec_field.name) != spec_field.default
        }
        raw: dict[str, object] = {
            "out": str(self.out_path),
            "params": environment,
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
            "params:\n"
            "  gamma: 0.999\n"
            "  horizon: 100\n"
            "  sigma: 1.0\n"
            "  effect: 0.0\n"
            "  effect_std: 0.5\n"
            "  size: 64\n"
            "  arms: 2\n"
            "workers: 4\n"
            "contestants:\n"
            "  ts: ProbabilityMatching\n"
            "  koth_half: {strategy: Koth3, sigma_factor: 0.5}\n"
        )
        sweep = ArenaSpec.load(root / "sweep.yaml")

        assert sweep.params.horizon == 100 and sweep.params.eta == 0.0
        assert sweep.workers == 4
        assert sweep.contestants["ts"] == {"strategy": "ProbabilityMatching"}
        assert sweep.contestants["koth_half"] == {
            "strategy": "Koth3",
            "sigma_factor": 0.5,
        }

        sweep.save(root / "back.yaml")
        written = (root / "back.yaml").read_text()

        assert ArenaSpec.load(root / "back.yaml") == sweep
        assert "effect_std" in written and "workers" in written
        assert "eta" not in written and "device" not in written

        (root / "bad.yaml").write_text(
            "out: x.pkl\nreps: 5\n"
            "params: {gamma: 0.9, horizon: 1, sigma: 1.0, effect: 0.0,"
            " effect_std: 0.0, size: 1, arms: 2}\n"
        )

        try:
            ArenaSpec.load(root / "bad.yaml")

            raise AssertionError("unknown arena keys must raise")
        except ValueError as e:
            assert "reps" in str(e)

        (root / "bad_params.yaml").write_text(
            "out: x.pkl\n"
            "params: {gamma: 0.9, horizon: 1, sigma: 1.0, effect: 0.0,"
            " effect_std: 0.0, size: 1, arms: 2, batch: 7}\n"
        )

        try:
            ArenaSpec.load(root / "bad_params.yaml")

            raise AssertionError("unknown params keys must raise")
        except TypeError as e:
            assert "batch" in str(e)
    print("ok")
