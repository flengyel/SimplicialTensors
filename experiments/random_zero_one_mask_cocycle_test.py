"""Check boundary linearity and boundary cycles for symbolic masked matrices.

The historical filename uses "cocycle". These are chain-boundary checks over
F_2 by default, with an optional signed calculation over Z.
"""

import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import sympy as sp

from simplicial_tensors.symbolic_tensor_ops import SymbolicTensor
from simplicial_tensors.tensor_ops import bdry


def permutation_cycle_partition(perm):
    if sorted(perm) != list(range(len(perm))):
        raise ValueError("perm must be a permutation of range(n)")
    seen, partition = set(), []
    for i in range(len(perm)):
        if i in seen:
            continue
        j, length = i, 0
        while j not in seen:
            seen.add(j)
            length += 1
            j = perm[j]
        partition.append(length)
    return tuple(sorted(partition, reverse=True))


def random_permutation(n, rng=None):
    """Sample from a supplied NumPy generator, or a local generator."""
    rng = np.random.default_rng() if rng is None else rng
    return rng.permutation(n).tolist()


def kneser_petersen_adjacency():
    vertices = list(combinations(range(1, 6), 2))
    return np.array([[int(set(u).isdisjoint(v)) for v in vertices]
                     for u in vertices], dtype=int)


def random_zero_one_matrix(n, density=0.5, seed=None, rng=None):
    """Sample an n by n mask without changing NumPy's global random state."""
    if n < 1 or not 0 <= density <= 1:
        raise ValueError("n must be positive and density in [0, 1]")
    if rng is not None and seed is not None:
        raise ValueError("supply either seed or rng")
    rng = np.random.default_rng(seed) if rng is None else rng
    return (rng.random((n, n)) < density).astype(int)


def _reduce_tensor(tensor, modulus):
    if modulus not in (None, 2):
        raise ValueError("modulus must be None (signed) or 2")
    result = np.empty(tensor.shape, dtype=object)
    for index in np.ndindex(tensor.shape):
        expression = sp.expand(tensor[index])
        if modulus == 2:
            symbols = sorted(expression.free_symbols, key=str)
            expression = (sp.Poly(expression, *symbols, modulus=2).as_expr()
                          if symbols else sp.Integer(expression) % 2)
        result[index] = expression
    return result


def boundary_difference_check(source_a, source_b, modulus=2):
    """Check D = boundary(source_a - source_b) and boundary(D) = 0.

    Both sources have the same matrix shape, with at least two rows and
    columns. The returned boundary arrays retain their coordinate positions.
    """
    source_a = np.asarray(source_a, dtype=object)
    source_b = np.asarray(source_b, dtype=object)
    if (source_a.shape != source_b.shape or source_a.ndim != 2
            or min(source_a.shape) < 2):
        raise ValueError("sources must have the same matrix shape with axes at least 2")
    boundary_a = _reduce_tensor(bdry(source_a), modulus)
    boundary_b = _reduce_tensor(bdry(source_b), modulus)
    difference = _reduce_tensor(boundary_a - boundary_b, modulus)
    source_difference = _reduce_tensor(source_a - source_b, modulus)
    residual = _reduce_tensor(difference - bdry(source_difference), modulus)
    next_boundary = _reduce_tensor(bdry(difference), modulus)
    is_boundary = all(entry == 0 for entry in residual.flat)
    is_cycle = all(entry == 0 for entry in next_boundary.flat)
    return {
        "boundary_a": boundary_a,
        "boundary_b": boundary_b,
        "boundary_difference": difference,
        "nonzero_entries": sum(entry != 0 for entry in difference.flat),
        "is_boundary": is_boundary,
        "is_cycle": is_cycle,
        "ok": is_boundary and is_cycle,
    }


def _report(results, label):
    passed = sum(result["ok"] for result in results)
    print(f"{label}: {passed}/{len(results)} boundary and cycle checks passed.")


def test_random_zero_one_boundaries(n=10, num_trials=20, density=0.5,
                                    seed=123, modulus=2):
    if n < 2 or num_trials < 1:
        raise ValueError("n must be at least 2 and num_trials positive")
    rng = np.random.default_rng(seed)
    tensor = SymbolicTensor((n, n)).tensor
    results = []
    for trial in range(num_trials):
        source_a = tensor * random_zero_one_matrix(n, density, rng=rng)
        source_b = tensor * random_zero_one_matrix(n, density, rng=rng)
        result = boundary_difference_check(source_a, source_b, modulus)
        result["trial"] = trial
        results.append(result)
        print(f"Mask trial {trial:02d}: {'PASS' if result['ok'] else 'FAIL'}, "
              f"{result['nonzero_entries']} nonzero boundary entries")
    _report(results, "Random masks")
    return results


def test_petersen_permutation_boundaries(n=10, num_trials=20, seed=123,
                                       modulus=2):
    if n != 10 or num_trials < 1:
        raise ValueError("the Petersen pattern has n=10 and num_trials must be positive")
    rng = np.random.default_rng(seed)
    source_a = SymbolicTensor((n, n)).tensor * kneser_petersen_adjacency()
    results = []
    for trial in range(num_trials):
        perm = random_permutation(n, rng)
        # perm lists the old labels in the new row and column order.
        source_b = source_a[np.ix_(perm, perm)]
        result = boundary_difference_check(source_a, source_b, modulus)
        result.update(trial=trial, partition=permutation_cycle_partition(perm))
        results.append(result)
        print(f"Petersen trial {trial:02d}: {'PASS' if result['ok'] else 'FAIL'}, "
              f"{result['nonzero_entries']} nonzero boundary entries, "
              f"partition {result['partition']}")
    _report(results, "Petersen relabelings")
    print("Permutation cycle-partition frequencies:")
    for partition, count in sorted(Counter(r["partition"] for r in results).items()):
        print(f"{partition}: {count}")
    return results


def test_random_zero_one_cocycles(n=10, num_trials=20, density=0.5,
                                 seed=123, modulus=2):
    """Compatibility entry point for the random-mask boundary checks."""
    return test_random_zero_one_boundaries(n, num_trials, density, seed, modulus)


def test_petersen_permutation_cocycles(n=10, num_trials=20, seed=123,
                                     modulus=2):
    """Compatibility entry point for the Petersen boundary checks."""
    return test_petersen_permutation_boundaries(n, num_trials, seed, modulus)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--masks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--signed", action="store_true", help="work over Z instead of F_2")
    args = parser.parse_args(argv)
    if min(args.permutations, args.masks) < 1 or not 0 <= args.density <= 1:
        parser.error("trial counts must be positive and density in [0, 1]")
    modulus = None if args.signed else 2
    print(f"seed={args.seed}, coefficients={'Z' if args.signed else 'F_2'}")
    results = test_petersen_permutation_boundaries(
        num_trials=args.permutations, seed=args.seed, modulus=modulus)
    results += test_random_zero_one_boundaries(
        num_trials=args.masks, density=args.density, seed=args.seed + 1, modulus=modulus)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
