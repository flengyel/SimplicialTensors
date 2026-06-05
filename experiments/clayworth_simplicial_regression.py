#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    clayworth_simplicial_regression.py
#
#    Copyright (C) 2026 Florian Lengyel
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Clayworth source-material notice (attribution for claims under review,
#    not a license statement for this GPL experiment):
#
#      © 2026 Logocentricity Inc.
#      U.S. Provisional Patent Application 63/961,154.
#      Released for academic citation, peer review, and scientific verification.
#      Paul Clayworth
#      Founder & CEO
#      Logocentricity Inc.
#      Capital Factory, Austin, TX
#
#    This experiment is an independent regression/stress test of finite
#    mathematical consequences of the Clayworth simplicial-object claims. It
#    does not reproduce the Clayworth source document or assert ownership of
#    Clayworth Algebra Framework text, terminology, trademarks, or patent claims.

"""Finite regression tests for Clayworth-style simplicial-object claims.

This experiment uses the SimplicialTensors tensor API as the reference
implementation for standard face, degeneracy, horn, and filler operations.
It then tests finite consequences of the Clayworth document's simplicial
claims on small finite models.

The tests are intentionally modest: they do not try to prove or disprove a
topos equivalence. They check local finite implications that should hold
before such an equivalence can be taken seriously.

A PASS means the expected regression condition was observed. For refutation
tests, this means the expected finite counterexample was found.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Callable

# Allow direct execution from either the repository root or experiments/.
# This keeps the script usable even before `pip install -e .` has been run.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from simplicial_tensors.tensor_ops import (
    degen,
    dimen,
    face,
    filler,
    horn,
    kan_condition,
    range_tensor,
)


FANO_LINES: tuple[frozenset[int], ...] = (
    frozenset({1, 2, 3}),
    frozenset({1, 4, 5}),
    frozenset({1, 7, 6}),
    frozenset({2, 4, 6}),
    frozenset({2, 5, 7}),
    frozenset({3, 4, 7}),
    frozenset({3, 6, 5}),
)
FANO_POINTS = frozenset(range(1, 8))


@dataclass(frozen=True)
class RegressionResult:
    """One finite regression result."""

    name: str
    passed: bool
    detail: str


def tuple_face(simplex: tuple[int, ...], i: int) -> tuple[int, ...]:
    """Delete the i-th vertex of a finite tuple simplex."""

    return simplex[:i] + simplex[i + 1 :]


def tuple_degen(simplex: tuple[int, ...], i: int) -> tuple[int, ...]:
    """Repeat the i-th vertex of a finite tuple simplex."""

    return simplex[: i + 1] + (simplex[i],) + simplex[i + 1 :]


def is_fano_closed(subset: frozenset[int]) -> bool:
    """Fano closure rule used in the Clayworth text."""

    return all(
        not (len(line & subset) >= 2 and not line.issubset(subset))
        for line in FANO_LINES
    )


def enumerate_fano_closed_objects() -> tuple[frozenset[int], ...]:
    """Enumerate the 16 Fano-closed subsets."""

    objects: list[frozenset[int]] = []
    points = tuple(sorted(FANO_POINTS))
    for mask in range(1 << len(points)):
        subset = frozenset(points[i] for i in range(len(points)) if mask & (1 << i))
        if is_fano_closed(subset):
            objects.append(subset)
    return tuple(sorted(objects, key=lambda s: (len(s), tuple(sorted(s)))))


def leq(left: frozenset[int], right: frozenset[int]) -> bool:
    """Order relation in the Fano closed-subset poset."""

    return left.issubset(right)


def check_tensor_simplicial_identities() -> RegressionResult:
    """Check ordinary simplicial identities using SimplicialTensors operations."""

    shapes = ((3, 3), (4, 4, 4), (5, 6, 5))
    checked = 0

    for shape in shapes:
        tensor = range_tensor(shape)
        n_faces = min(tensor.shape)

        # d_i d_j = d_{j-1} d_i for i < j.
        for i in range(n_faces):
            for j in range(i + 1, n_faces):
                lhs = face(face(tensor, j), i)
                rhs = face(face(tensor, i), j - 1)
                checked += 1
                if not np.array_equal(lhs, rhs):
                    return RegressionResult(
                        "SimplicialTensors face-face identities",
                        False,
                        f"failed for shape={shape}, i={i}, j={j}",
                    )

        # s_i s_j = s_{j+1} s_i for i <= j.
        for i in range(n_faces):
            for j in range(i, n_faces):
                lhs = degen(degen(tensor, j), i)
                rhs = degen(degen(tensor, i), j + 1)
                checked += 1
                if not np.array_equal(lhs, rhs):
                    return RegressionResult(
                        "SimplicialTensors degeneracy-degeneracy identities",
                        False,
                        f"failed for shape={shape}, i={i}, j={j}",
                    )

        # Mixed identity:
        # d_i s_j = s_{j-1} d_i for i < j,
        # d_i s_j = id for i = j or i = j+1,
        # d_i s_j = s_j d_{i-1} for i > j+1.
        for j in range(n_faces):
            degenerated = degen(tensor, j)
            for i in range(min(degenerated.shape)):
                lhs = face(degenerated, i)
                if i < j:
                    rhs = degen(face(tensor, i), j - 1)
                elif i in {j, j + 1}:
                    rhs = tensor
                else:
                    rhs = degen(face(tensor, i - 1), j)
                checked += 1
                if not np.array_equal(lhs, rhs):
                    return RegressionResult(
                        "SimplicialTensors mixed face-degeneracy identities",
                        False,
                        f"failed for shape={shape}, i={i}, j={j}",
                    )

    return RegressionResult(
        "SimplicialTensors simplicial identities",
        True,
        f"checked {checked} face/degeneracy identities on {len(shapes)} shapes",
    )


def check_tensor_horn_pipeline() -> RegressionResult:
    """Check horn compatibility and Moore filler round-trips."""

    shapes = ((3, 3), (4, 4, 4), (5, 6, 5))
    checked = 0

    for shape in shapes:
        tensor = range_tensor(shape)
        for omitted in range(dimen(tensor) + 1):
            h = horn(tensor, omitted)
            checked += 1

            if not kan_condition(h, omitted):
                return RegressionResult(
                    "SimplicialTensors horn compatibility",
                    False,
                    f"kan_condition failed for shape={shape}, omitted={omitted}",
                )

            filled = filler(h, omitted)
            reconstructed_horn = horn(filled, omitted)
            if not np.array_equal(h, reconstructed_horn):
                return RegressionResult(
                    "SimplicialTensors horn/filler round-trip",
                    False,
                    f"filler horn mismatch for shape={shape}, omitted={omitted}",
                )

    return RegressionResult(
        "SimplicialTensors horn/filler round-trip",
        True,
        f"checked {checked} horns across {len(shapes)} tensor shapes",
    )


def check_all_tuple_model(max_dim: int, state_count: int) -> RegressionResult:
    """Positive control: all (n+1)-tuples form a simplicial set."""

    states = tuple(range(state_count))
    checked = 0

    for n in range(max_dim + 1):
        for simplex in product(states, repeat=n + 1):
            simplex = tuple(simplex)

            for i in range(n + 1):
                f = tuple_face(simplex, i)
                checked += 1
                if len(f) != n:
                    return RegressionResult(
                        "All-tuples model closure under faces",
                        False,
                        f"bad face length for simplex={simplex}, i={i}",
                    )

            for i in range(n + 1):
                s = tuple_degen(simplex, i)
                checked += 1
                if len(s) != n + 2:
                    return RegressionResult(
                        "All-tuples model closure under degeneracies",
                        False,
                        f"bad degeneracy length for simplex={simplex}, i={i}",
                    )

            if n >= 2:
                for i in range(n + 1):
                    for j in range(i + 1, n + 1):
                        lhs = tuple_face(tuple_face(simplex, j), i)
                        rhs = tuple_face(tuple_face(simplex, i), j - 1)
                        checked += 1
                        if lhs != rhs:
                            return RegressionResult(
                                "All-tuples model face identities",
                                False,
                                f"failed for simplex={simplex}, i={i}, j={j}",
                            )

    return RegressionResult(
        "All-tuples model is simplicial but dynamics-free",
        True,
        (
            f"checked {checked} finite tuple operations through dimension {max_dim}; "
            "this validates only the trivial all-tuples construction"
        ),
    )


def check_stage1_lambda21_is_trivial(state_count: int) -> RegressionResult:
    """Replicate the Stage-1-style Lambda^2_1 filler check on all tuples."""

    states = tuple(range(state_count))
    checked = 0

    for s0, s1, s2 in product(states, repeat=3):
        filler_simplex = (s0, s1, s2)
        horn_01 = (s0, s1)
        horn_12 = (s1, s2)
        checked += 1
        if tuple_face(filler_simplex, 2) != horn_01:
            return RegressionResult(
                "Stage-1-style Lambda^2_1 tuple filler",
                False,
                f"first horn edge mismatch for filler={filler_simplex}",
            )
        if tuple_face(filler_simplex, 0) != horn_12:
            return RegressionResult(
                "Stage-1-style Lambda^2_1 tuple filler",
                False,
                f"second horn edge mismatch for filler={filler_simplex}",
            )

    return RegressionResult(
        "Stage-1-style Lambda^2_1 check is tautological on all tuples",
        True,
        (
            f"filled {checked} horns by the tuple (s0,s1,s2); "
            "no Phi-compatibility or higher-dimensional condition is tested"
        ),
    )


def strict_phi(x: int) -> int:
    """A finite non-identity update map used to model Phi-orbit segments."""

    return max(x - 1, 0)


def is_strict_phi_simplex(
    simplex: tuple[int, ...],
    phi: Callable[[int], int] = strict_phi,
) -> bool:
    """A strict Phi-simplex has consecutive vertices s_{i+1}=Phi(s_i)."""

    return all(simplex[i + 1] == phi(simplex[i]) for i in range(len(simplex) - 1))


def strict_phi_simplices(
    states: tuple[int, ...],
    n: int,
    phi: Callable[[int], int] = strict_phi,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate strict Phi-orbit n-simplices over a finite state set."""

    return tuple(
        tuple(simplex)
        for simplex in product(states, repeat=n + 1)
        if is_strict_phi_simplex(tuple(simplex), phi)
    )


def check_strict_phi_faces_fail(max_dim: int, state_count: int) -> RegressionResult:
    """Find a face-closure counterexample for strict Phi-orbit tuples."""

    states = tuple(range(state_count))

    for n in range(2, max_dim + 1):
        for simplex in strict_phi_simplices(states, n):
            for i in range(1, n):
                f = tuple_face(simplex, i)
                if not is_strict_phi_simplex(f):
                    return RegressionResult(
                        "Strict Phi-orbit tuples are not face-closed",
                        True,
                        (
                            f"counterexample: simplex={simplex}, d_{i}={f}; "
                            "deleting an intermediate vertex creates a Phi-gap"
                        ),
                    )

    return RegressionResult(
        "Strict Phi-orbit tuples are not face-closed",
        False,
        f"no counterexample found through dimension {max_dim}",
    )


def check_strict_phi_degeneracies_fail(max_dim: int, state_count: int) -> RegressionResult:
    """Find a degeneracy-closure counterexample for strict Phi-orbit tuples."""

    states = tuple(range(state_count))

    for n in range(1, max_dim + 1):
        for simplex in strict_phi_simplices(states, n):
            for i in range(n + 1):
                s = tuple_degen(simplex, i)
                if not is_strict_phi_simplex(s):
                    return RegressionResult(
                        "Strict Phi-orbit tuples are not degeneracy-closed",
                        True,
                        (
                            f"counterexample: simplex={simplex}, s_{i}={s}; "
                            "a repeated non-fixed vertex is not a Phi-step"
                        ),
                    )

    return RegressionResult(
        "Strict Phi-orbit tuples are not degeneracy-closed",
        False,
        f"no counterexample found through dimension {max_dim}",
    )


def check_fano_closed_object_count() -> RegressionResult:
    """Check the finite Fano site object count."""

    objects = enumerate_fano_closed_objects()
    counts = {
        "empty": sum(len(obj) == 0 for obj in objects),
        "points": sum(len(obj) == 1 for obj in objects),
        "lines": sum(obj in FANO_LINES for obj in objects),
        "full": sum(obj == FANO_POINTS for obj in objects),
    }
    expected = {"empty": 1, "points": 7, "lines": 7, "full": 1}
    passed = len(objects) == 16 and counts == expected
    return RegressionResult(
        "Fano-closed subsets have the advertised 1+7+7+1 count",
        passed,
        f"count={len(objects)}, partition={counts}",
    )


def check_fano_poset_inner_2_horns() -> RegressionResult:
    """Check inner Lambda^2_1 horns in the Fano closed-subset poset nerve."""

    objects = enumerate_fano_closed_objects()
    checked = 0

    for x0, x1, x2 in product(objects, repeat=3):
        if leq(x0, x1) and leq(x1, x2):
            checked += 1
            if not leq(x0, x2):
                return RegressionResult(
                    "Fano poset nerve inner 2-horns fill",
                    False,
                    f"transitivity failed for {x0} <= {x1} <= {x2}",
                )

    return RegressionResult(
        "Fano poset nerve inner 2-horns fill",
        True,
        f"checked {checked} composable edge pairs; fillers exist by transitivity",
    )


def check_fano_poset_not_kan() -> RegressionResult:
    """Exhibit an outer horn obstruction in the Fano closed-subset poset nerve."""

    point = frozenset({1})
    line_a = frozenset({1, 2, 3})
    line_b = frozenset({1, 4, 5})

    horn_edges_exist = leq(point, line_a) and leq(point, line_b)
    no_outer_filler = not leq(line_a, line_b) and not leq(line_b, line_a)

    if horn_edges_exist and no_outer_filler:
        return RegressionResult(
            "Fano poset nerve is not Kan",
            True,
            (
                "outer 2-horn with edges {1}<={1,2,3} and {1}<={1,4,5} "
                "has no filler because the two lines are incomparable"
            ),
        )

    return RegressionResult(
        "Fano poset nerve is not Kan",
        False,
        "expected incomparable-line outer horn was not found",
    )


def run_regression(max_dim: int, state_count: int) -> list[RegressionResult]:
    """Run the finite regression suite."""

    return [
        check_tensor_simplicial_identities(),
        check_tensor_horn_pipeline(),
        check_all_tuple_model(max_dim=max_dim, state_count=state_count),
        check_stage1_lambda21_is_trivial(state_count=state_count),
        check_strict_phi_faces_fail(max_dim=max_dim, state_count=state_count),
        check_strict_phi_degeneracies_fail(max_dim=max_dim, state_count=state_count),
        check_fano_closed_object_count(),
        check_fano_poset_inner_2_horns(),
        check_fano_poset_not_kan(),
    ]


def print_table(results: list[RegressionResult]) -> None:
    """Print a compact text table."""

    print("=" * 100)
    print("Clayworth simplicial-object finite regression")
    print("=" * 100)
    print(f"{'Status':<8} {'Check':<62} Detail")
    print("-" * 100)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:<8} {result.name:<62} {result.detail}")
    print("-" * 100)
    n_pass = sum(result.passed for result in results)
    print(f"Result: {n_pass}/{len(results)} expected finite conditions observed.")
    print()
    print("Interpretation:")
    print("  * The all-tuples construction satisfies the simplicial identities,")
    print("    but that is the dynamics-free/codiscrete case.")
    print("  * The strict Phi-orbit interpretation is not a simplicial set in")
    print("    this finite model, because faces and degeneracies are not closed.")
    print("  * The Fano poset nerve has inner 2-horn fillers, but it is not Kan.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Finite regression tests for Clayworth-style simplicial claims."
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=4,
        help="maximum finite tuple dimension to test",
    )
    parser.add_argument(
        "--state-count",
        type=int,
        default=4,
        help="number of finite states used in tuple/Phi models",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of the text table",
    )
    args = parser.parse_args()

    if args.max_dim < 2:
        parser.error("--max-dim must be at least 2")
    if args.state_count < 3:
        parser.error("--state-count must be at least 3")

    results = run_regression(max_dim=args.max_dim, state_count=args.state_count)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_table(results)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
