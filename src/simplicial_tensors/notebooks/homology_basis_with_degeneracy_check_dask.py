"""Coordinate-unit diagnostics and exact relative homology examples.

This module requires no Dask and executes examples only when run as a module.
"""

import numpy as np

from ..homology import relative_horn_homology
from ..tensor_ops import is_degen


def standard_basis_tensor(idx, shape):
    """Return the coordinate tensor E_idx of the supplied shape."""
    tensor = np.zeros(shape, dtype=object)
    tensor[idx] = 1
    return tensor


def get_non_degenerate_indices(shape):
    """Return coordinate-unit indices whose tensors lie outside the degeneracy span.

    Their images need not be linearly independent in the normalized module.
    """
    return {index for index in np.ndindex(shape)
            if not is_degen(standard_basis_tensor(index, shape))}


def demonstrate_degeneracy(shape):
    """Print coordinate-unit counts, without identifying them with dimensions."""
    total = int(np.prod(shape))
    count = len(get_non_degenerate_indices(shape))
    print(f"shape={shape}, coordinate units={total}, "
          f"units in degeneracy span={total - count}, units outside span={count}")


def compute_homology_dimension(shape, j=0, *, tensor=None, seed=123, modulus=None):
    """Return top homology relative to the submodule of included horn faces.

    Use the supplied tensor, or fixed-seed integer entries in -5,...,5.
    The coefficient field is Q unless a prime modulus is supplied.
    """
    shape = tuple(shape)
    if tensor is None:
        tensor = np.random.default_rng(seed).integers(-5, 6, size=shape)
    elif np.shape(tensor) != shape:
        raise ValueError("The supplied tensor must have the stated shape.")
    result = relative_horn_homology(tensor, j, modulus=modulus)
    degree = min(shape) - 1
    return result.homology_dimensions[degree] if degree < len(result.homology_dimensions) else 0


def main():
    for shape in [(2, 2), (3, 3), (2, 2, 2)]:
        demonstrate_degeneracy(shape)
        dimension = compute_homology_dimension(shape)
        print(f"shape={shape}, seed=123, omitted face=0, top relative Betti number={dimension}")


if __name__ == "__main__":
    main()
