#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    n_hypergroupoid_conjecture.py
#
#    Copyright (C) 2021-2025 Florian Lengyel
#    Email: florian.lengyel at cuny edu, florian.lengyel at gmail
#    Web: https://github.com/flengyel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Check full-module horn uniqueness on reproducible random tensors."""

import argparse

import numpy as np

from simplicial_tensors.tensor_ops import (
    ___SEED___,
    bdry,
    degen,
    dimen,
    face,
    is_degen,
    n_hypergroupoid_comparison,
    n_hypergroupoid_conjecture,
    order,
    random_tensor,
)


def random_shape(n, *, rng=None):
    """Sample a shape with 2 through n//2 axes and lengths 3 through n."""
    if not isinstance(n, (int, np.integer)) or n < 4:
        raise ValueError("n must be an integer at least four")
    rng = np.random.default_rng(___SEED___) if rng is None else rng
    axes = int(rng.integers(2, n // 2 + 1))
    return tuple(int(value) for value in rng.integers(3, n + 1, size=axes))


def _attempt_limit(max_attempts):
    if not isinstance(max_attempts, (int, np.integer)) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    return max_attempts


def _sample_tensor(shape, rng):
    seed = int(rng.integers(0, np.iinfo(np.int64).max))
    return random_tensor(shape, low=1, high=10, seed=seed)


def _check_nondegenerate_boundary_possible(shape):
    degree = min(shape) - 1
    if degree > len(shape):
        raise ValueError(
            f"Cannot sample a nondegenerate boundary for shape {shape}: "
            f"simplicial degree {degree} exceeds {len(shape)} axes"
        )


def generate_tensor(shape, force_degeneracy, skip_degeneracies, *, max_attempts=100, rng=None):
    """Sample a tensor, with at most max_attempts boundary rejections.

    Forced degeneracy applies s_0 once and takes precedence over the
    optional rejection of degenerate boundaries.
    """
    max_attempts = _attempt_limit(max_attempts)
    if skip_degeneracies and not force_degeneracy:
        _check_nondegenerate_boundary_possible(shape)
    rng = np.random.default_rng(___SEED___) if rng is None else rng
    for _ in range(max_attempts):
        tensor = _sample_tensor(shape, rng)
        if force_degeneracy:
            return degen(tensor, 0)
        if not skip_degeneracies or not is_degen(bdry(tensor)):
            return tensor
    raise RuntimeError(f"No tensor with a nondegenerate boundary in {max_attempts} attempts for shape {shape}")


def run_n_hypergroupoid_conjecture(
    tests,
    maxdim,
    force_degeneracy=False,
    skip_degeneracies=False,
    outer_horns=False,
    single_shape=(0, 0),
    *,
    max_attempts=100,
    seed=___SEED___,
    verbose=True,
):
    """Compare checked horn uniqueness with the shape criterion.

    ``single_shape=None`` and the legacy sentinel ``(0, 0)`` select random
    shapes. The prediction always uses the sampled tensor's final shape.
    """
    if not isinstance(tests, (int, np.integer)) or tests < 0:
        raise ValueError("tests must be a nonnegative integer")
    _attempt_limit(max_attempts)
    rng = np.random.default_rng(seed)
    random_shapes = single_shape is None or tuple(single_shape) == (0, 0)
    if not random_shapes and (not single_shape or min(single_shape) < 3):
        raise ValueError("shape lengths must be at least three")
    unique = 0
    for trial in range(tests):
        shape = random_shape(maxdim, rng=rng) if random_shapes else tuple(single_shape)
        tensor = generate_tensor(
            shape, force_degeneracy, skip_degeneracies,
            max_attempts=max_attempts, rng=rng,
        )
        predicted = n_hypergroupoid_conjecture(tensor.shape, verbose=verbose)
        comparison = n_hypergroupoid_comparison(
            tensor, outer_horns=outer_horns, verbose=verbose, allow_degen=True,
        )
        if comparison != predicted:
            print(f"Mismatch at trial {trial + 1}, shape {tensor.shape}, seed {seed}")
            print(f"Shape criterion={predicted}, checked uniqueness={comparison}")
            return False
        unique += bool(comparison)
    print(f"Checked {tests} tensors with seed {seed}.")
    print(f"Tensors with unique checked horn fillers: {unique}")
    print(f"Tensors with nonunique checked horn fillers: {tests - unique}")
    return True


def verify_unique_down_to_order(
    shape=(4, 4, 4), outer_horns=False, *, max_attempts=100,
    seed=___SEED___, verbose=True,
):
    """Check the degree threshold after three degeneracies and successive d_0."""
    max_attempts = _attempt_limit(max_attempts)
    _check_nondegenerate_boundary_possible(shape)
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        tensor = _sample_tensor(shape, rng)
        if not is_degen(tensor) and not is_degen(bdry(tensor)):
            break
    else:
        raise RuntimeError(f"No tensor with nondegenerate tensor and boundary in {max_attempts} attempts for shape {shape}")

    for _ in range(3):
        tensor = degen(tensor, 0)
    initial_shape = tensor.shape
    while True:
        predicted = n_hypergroupoid_conjecture(tensor.shape, verbose=verbose)
        comparison = n_hypergroupoid_comparison(
            tensor, outer_horns=outer_horns, verbose=verbose, allow_degen=True,
        )
        if comparison != predicted:
            print(f"Degree-threshold mismatch for shape {tensor.shape}, seed {seed}")
            return False
        if not predicted:
            print(f"Checked descent from shape {initial_shape} to degree {dimen(tensor)} with {order(tensor)} axes.")
            return True
        tensor = face(tensor, 0)


def _shape_argument(value):
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated axis lengths") from exc
    if not shape or min(shape) < 3:
        raise argparse.ArgumentTypeError("axis lengths must be at least three")
    return shape


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", type=int, default=10)
    parser.add_argument("--maxdim", type=int, default=6, help="maximum axis length for random shapes")
    shapes = parser.add_mutually_exclusive_group()
    shapes.add_argument("--shape", type=_shape_argument, default=(3, 3))
    shapes.add_argument("--random-shapes", action="store_true")
    parser.add_argument("--force-degeneracy", action="store_true")
    parser.add_argument("--skip-degeneracies", action="store_true")
    parser.add_argument("--outer-horns", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=___SEED___)
    parser.add_argument("--verify-descent", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    try:
        success = run_n_hypergroupoid_conjecture(
            args.tests, args.maxdim,
            force_degeneracy=args.force_degeneracy,
            skip_degeneracies=args.skip_degeneracies,
            outer_horns=args.outer_horns,
            single_shape=None if args.random_shapes else args.shape,
            max_attempts=args.max_attempts, seed=args.seed, verbose=args.verbose,
        )
        if success and args.verify_descent:
            success = verify_unique_down_to_order(
                shape=args.shape, outer_horns=args.outer_horns,
                max_attempts=args.max_attempts, seed=args.seed, verbose=args.verbose,
            )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
