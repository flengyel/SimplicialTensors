"""Check cycle deletion on permutation matrices."""

import argparse
import itertools

import numpy as np


def perm_matrix(perm):
    """Return the matrix with entry one at ``(i, perm[i])``."""
    perm = tuple(perm)
    if sorted(perm) != list(range(len(perm))):
        raise ValueError("perm must be a permutation of range(len(perm))")
    matrix = np.zeros((len(perm), len(perm)), dtype=int)
    for i, j in enumerate(perm):
        matrix[i, j] = 1
    return matrix


def face_matrix(matrix, i):
    """Delete element ``i`` from its permutation cycle and relabel survivors.

    For a permutation matrix P, delete row and column i from
    P + (P e_i)(e_i.T P). This is cycle deletion, rather than the package's
    principal-deletion face on arbitrary matrices.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 <= i < matrix.shape[0]:
        raise IndexError("deletion index is out of range")
    corrected = matrix + matrix[:, i:i + 1] @ matrix[i:i + 1, :]
    return np.delete(np.delete(corrected, i, axis=0), i, axis=1)


def is_perm(matrix):
    matrix = np.asarray(matrix)
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] == matrix.shape[1]
        and (matrix.sum(axis=0) == 1).all()
        and (matrix.sum(axis=1) == 1).all()
        and np.logical_or(matrix == 0, matrix == 1).all()
    )


def check_permutation_faces(size=7, *, verbose=False):
    """Check every deletion of every permutation on ``size`` elements."""
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("size must be a positive integer")
    checked = 0
    for perm in itertools.permutations(range(size)):
        matrix = perm_matrix(perm)
        for i in range(size):
            deleted = face_matrix(matrix, i)
            if not is_perm(deleted):
                raise AssertionError(f"Failed at permutation={perm}, index={i}")
            if verbose:
                print(f"permutation={perm}, index={i}, result={deleted.tolist()}")
            checked += 1
    return checked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=7, help="number of elements (default: 7)")
    parser.add_argument("--verbose", action="store_true", help="print each checked deletion")
    args = parser.parse_args(argv)
    if args.size < 1:
        parser.error("--size must be positive")
    checked = check_permutation_faces(args.size, verbose=args.verbose)
    print(f"Checked {checked} cycle deletions on permutations of size {args.size}.")


if __name__ == "__main__":
    main()
