"""Verify the deletion-boundary formula under simultaneous vertex relabeling."""

import numpy as np
from simplicial_tensors.tensor_ops import bdry, face


def _permutation(sigma):
    sigma = np.asarray(sigma)
    if (sigma.ndim != 1 or not np.issubdtype(sigma.dtype, np.integer)
            or sorted(sigma.tolist()) != list(range(len(sigma)))):
        raise ValueError("sigma must be a permutation of range(n)")
    if len(sigma) == 0:
        raise ValueError("sigma must be nonempty")
    return sigma


def perm_matrix_from_sigma(sigma):
    """Return P with P[sigma[j], j] = 1 (old vertex j gets label sigma[j])."""
    sigma = _permutation(sigma)
    P = np.zeros((len(sigma), len(sigma)), dtype=int)
    P[sigma, np.arange(len(sigma))] = 1
    return P


def conjugate_by_sigma(A, sigma):
    """Return P A P.T, with P[sigma[j], j] = 1."""
    P = perm_matrix_from_sigma(sigma)
    A = np.asarray(A)
    if A.shape != P.shape:
        raise ValueError("A must be square and have one row per vertex")
    return P @ A @ P.T


def induced_subperm_matrix(sigma, i):
    """Map old surviving labels to new surviving labels after deleting new i.

    Rows are indexed by sorted new labels other than i, and columns by
    sorted old labels other than sigma^{-1}(i).
    """
    P = perm_matrix_from_sigma(sigma)
    if not 0 <= i < len(P):
        raise IndexError("deleted vertex is out of range")
    return np.delete(np.delete(P, i, axis=0), inverse_image(sigma, i), axis=1)


def principal_face(A, i):
    """Return the principal submatrix obtained by deleting index i."""
    return face(A, i)


def boundary_by_faces_via_sigma(A, sigma):
    """Return sum_i (-1)^i Q_i d_{sigma^{-1}(i)}(A) Q_i.T."""
    A = np.asarray(A)
    sigma = _permutation(sigma)
    if A.shape != (len(sigma), len(sigma)):
        raise ValueError("A must be square and have one row per vertex")
    n = len(sigma)
    acc = np.zeros((n - 1, n - 1), dtype=A.dtype)
    for i in range(n):
        Q = induced_subperm_matrix(sigma, i)
        term = Q @ principal_face(A, inverse_image(sigma, i)) @ Q.T
        acc = acc + term if i % 2 == 0 else acc - term
    return acc


def inverse_image(sigma, i):
    """Return sigma^{-1}(i)."""
    sigma = _permutation(sigma)
    if not 0 <= i < len(sigma):
        raise IndexError("vertex is out of range")
    return int(np.flatnonzero(sigma == i)[0])


def verify_boundary_equivariance_once(n=6, directed=True, mod2=False, seed=0):
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    sigma = rng.permutation(n)
    A = rng.integers(0, 2, size=(n, n), dtype=int)
    np.fill_diagonal(A, 0)
    if not directed:
        A = np.triu(A, 1)
        A = A + A.T
    lhs = bdry(conjugate_by_sigma(A, sigma))
    rhs = boundary_by_faces_via_sigma(A, sigma)
    if mod2:
        lhs, rhs = lhs % 2, rhs % 2
    return np.array_equal(lhs, rhs), A, sigma, lhs, rhs


def run_many_trials(trials=50, n=6, directed=True, mod2=False, seed=123):
    if trials < 1:
        raise ValueError("trials must be positive")
    for t in range(trials):
        ok, A, sigma, lhs, rhs = verify_boundary_equivariance_once(
            n=n, directed=directed, mod2=mod2, seed=seed + t
        )
        if not ok:
            print(f"Boundary formula failed at seed {seed + t}.")
            print("sigma =", sigma)
            print("A =\n", A)
            print("bdry(P A P.T) =\n", lhs)
            print("Relabeled face sum =\n", rhs)
            return False
    print(f"Verified {trials} trials (n={n}, directed={directed}, mod2={mod2}, seed={seed}).")
    return True


def main():
    signed = run_many_trials(trials=100, n=7, directed=True, mod2=False)
    mod2 = run_many_trials(trials=100, n=7, directed=True, mod2=True)
    return 0 if signed and mod2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
