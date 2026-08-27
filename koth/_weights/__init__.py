"""The exported nets and their fixtures, shipped inside the wheel as `.npz`."""

from __future__ import annotations

import numpy as np

from importlib import resources


def load(name: str) -> dict[str, np.ndarray]:
    """
    The arrays in `<name>.npz`, by name. The `*.check` fixtures are excluded from
    the wheel (pyproject), so they load from a repo checkout only.
    """
    resource = resources.files(__package__) / f"{name}.npz"

    if not resource.is_file():
        raise FileNotFoundError(
            f"{name}.npz is not shipped in the wheel; run the self-checks from a checkout"
        )

    with resources.as_file(resource) as path:
        with np.load(path) as file:
            return {key: file[key] for key in file.files}
