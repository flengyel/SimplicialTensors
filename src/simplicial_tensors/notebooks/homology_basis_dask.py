"""Exact relative-homology checks retained under the historical notebook filename."""

import numpy as np

from .homology_basis_with_degeneracy_check_dask import (
    compute_homology_dimension,
    standard_basis_tensor,
)


def full_homology_verification(shape, j=0, *, tensor=None, seed=123, modulus=None):
    """Return the exact top relative Betti number for the supplied tensor.

    If tensor is omitted, use fixed-seed integer input. This entry point
    delegates to the shared normalized quotient calculation. Coordinate-unit
    counts do not supply homology dimensions.
    """
    return compute_homology_dimension(shape, j, tensor=tensor, seed=seed, modulus=modulus)


def main():
    cube = np.arange(8, dtype=object).reshape(2, 2, 2)
    cases = [
        ("zero", np.zeros((3, 3), dtype=object), 0, 0),
        ("circle", standard_basis_tensor((0, 1), (2, 2)), 0, 1),
        ("cube", cube, 0, 0),
        ("cube", cube, 1, 1),
    ]
    for label, tensor, omitted, expected in cases:
        dimension = full_homology_verification(tensor.shape, omitted, tensor=tensor)
        if dimension != expected:
            raise RuntimeError(f"Unexpected relative homology dimension for {label}.")
        print(f"{label}, omitted face={omitted}, top relative Betti number={dimension}")


if __name__ == "__main__":
    main()
