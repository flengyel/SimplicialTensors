"""Exact rational homology examples using NumPy and SymPy; SageMath is optional."""

import argparse

import numpy as np
import sympy as sp

from simplicial_tensors.homology import relative_horn_homology
from simplicial_tensors.sagemath_compatible_tensor_ops import SymbolicTensor


def compute_homology_rank(diffs, n, *, chain_dimension=None):
    """Compute dim_Q H_n from exact differential matrices indexed by source.

    The dimension of C_n is inferred from d_n or d_(n+1). When both are
    absent it must be supplied explicitly, including in degree zero.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise TypeError("The homology degree must be an integer.")
    if n < 0:
        raise ValueError("The homology degree must be nonnegative.")
    outgoing = sp.Matrix(diffs[n]) if n in diffs else None
    incoming = sp.Matrix(diffs[n + 1]) if n + 1 in diffs else None
    dimensions = [matrix.cols if side == "outgoing" else matrix.rows
                  for side, matrix in [("outgoing", outgoing), ("incoming", incoming)]
                  if matrix is not None]
    if chain_dimension is not None:
        if (not isinstance(chain_dimension, (int, np.integer))
                or isinstance(chain_dimension, (bool, np.bool_)) or chain_dimension < 0):
            raise ValueError("The chain dimension must be a nonnegative integer.")
        dimensions.append(int(chain_dimension))
    if not dimensions:
        raise ValueError("The chain dimension is required when both differentials are absent.")
    if len(set(dimensions)) != 1:
        raise ValueError("The differential shapes and chain dimension do not agree.")
    matrices = [matrix for matrix in (outgoing, incoming) if matrix is not None]
    if any(not value.is_Rational for matrix in matrices for value in matrix):
        raise TypeError("Exact homology requires integer or rational matrix entries.")
    if n == 0 and outgoing is not None and any(outgoing):
        raise ValueError("The unaugmented differential out of degree zero must be zero.")
    if outgoing is not None and incoming is not None and any(outgoing * incoming):
        raise ValueError("Consecutive differentials must compose to zero.")
    return dimensions[0] - sum(matrix.rank() for matrix in matrices)


def compute_relative_homology_rank(tensor, j_horn_excluded_face, *, modulus=None):
    """Return top relative homology dimension for an exact tensor.

    Accept a NumPy array or a package SymbolicTensor whose entries are
    integers or rationals. Unevaluated symbolic coefficients are rejected.
    """
    array = np.asarray(getattr(tensor, "tensor", tensor), dtype=object)
    result = relative_horn_homology(array, j_horn_excluded_face, modulus=modulus)
    degree = min(array.shape) - 1
    return result.homology_dimensions[degree] if degree < len(result.homology_dimensions) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, help="Prime coefficient modulus; default Q")
    args = parser.parse_args()
    tensor = SymbolicTensor((2, 2, 2), tensor=np.arange(8).reshape(2, 2, 2))
    print(f"Coefficient field: {'Q' if args.modulus is None else f'F_{args.modulus}'}")
    for omitted in range(2):
        result = relative_horn_homology(tensor.tensor, omitted, modulus=args.modulus)
        print(f"omitted face={omitted}, relative Betti numbers={result.homology_dimensions}")


if __name__ == "__main__":
    main()
