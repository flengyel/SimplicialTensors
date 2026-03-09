"""Run a tiny A/B experiment comparing horn-filler uniqueness."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from simplicial_tensors.tensor_ops import (
    filler,
    horn,
    n_hypergroupoid_conjecture,
    random_tensor,
)


def run_case(shape: Tuple[int, ...], missing_face: int, seed: int) -> None:
    tensor = random_tensor(shape, low=-5, high=6, seed=seed)
    horn_faces = horn(tensor, missing_face)
    reconstructed = filler(horn_faces, missing_face)
    unique = np.array_equal(tensor, reconstructed)
    predicted = n_hypergroupoid_conjecture(shape)

    print(
        f"shape={shape} omitted_face={missing_face} predicted_unique={predicted} "
        f"observed_unique={unique}"
    )


def main() -> None:
    print("=== Horn filler A/B demo ===")
    run_case(shape=(3, 3), missing_face=1, seed=123)
    run_case(shape=(5, 5), missing_face=1, seed=456)


if __name__ == "__main__":
    main()
