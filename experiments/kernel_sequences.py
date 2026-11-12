"""Compute universal kernels, tensor realizations, and quotient lifts exactly."""

from __future__ import annotations

import argparse
import json

import numpy as np
import sympy as sp

from simplicial_tensors.chain_complex import ChainComplex, ExactField
from simplicial_tensors.chain_maps import (
    CapacityError, ChainMap, chain_embedding, chain_isomorphism,
    lift_quotient_isomorphism,
)
from simplicial_tensors.kernel_sequences import (
    kernel_sequence, realize_kernel, simplex_chain_complex,
)
from simplicial_tensors.rank_shifted import SparseTensor


CASES = ("jump", "marked", "spheres", "realize", "lifting", "obstruction")


def jump_tensor(parameter):
    return np.array([[0, 1, 0], [0, 1, 1], [1, 1, parameter]], dtype=object)


def _matrix(matrix):
    return {"shape": list(matrix.shape),
            "entries": [[str(value) for value in matrix.row(row)]
                        for row in range(matrix.rows)]}


def _summary(presentation, matrices=False):
    result = {
        "source_dimensions": presentation.source.dimensions,
        "kernel_dimensions": presentation.kernel.dimensions,
        "quotient_dimensions": presentation.quotient.dimensions,
        "boundary_ranks": presentation.quotient.homology().boundary_ranks,
        "betti_numbers": presentation.quotient.homology().homology_dimensions,
    }
    if matrices:
        result["source_bases"] = presentation.source.ambient_bases
        result["tensor_bases"] = presentation.tensor_bases
        result["realization_matrices"] = [_matrix(value) for value in presentation.g_matrices]
        result["kernel_inclusions"] = [_matrix(value) for value in presentation.kernel.lifts]
        result["kernel_differentials"] = [_matrix(value) for value in presentation.kernel.differentials]
    return result


def _same_kernels(left, right):
    if left.source.dimensions != right.source.dimensions:
        return False
    field = left.source.field
    return all(a.cols == b.cols == field.rank(a.row_join(b))
               for a, b in zip(left.kernel.lifts, right.kernel.lifts))


def _capacity_error(operation):
    try:
        operation()
    except CapacityError as error:
        return {"degree": error.degree, "kind": error.kind,
                "required": error.required, "available": error.available}
    raise AssertionError("The proposed embedding should fail a capacity condition.")


def examples(*, modulus=None, case="all", max_degree=3, parameter=4, matrices=False):
    field = ExactField(modulus)
    if case not in (*CASES, "all"):
        raise ValueError("Unknown example case.")
    if isinstance(max_degree, bool) or not isinstance(max_degree, int) or max_degree < 1:
        raise ValueError("The maximum sphere degree must be positive.")
    records = {}
    selected = CASES if case == "all" else (case,)
    if "jump" in selected:
        records["jump"] = {
            str(t): _summary(kernel_sequence(jump_tensor(t), modulus=modulus), matrices)
            for t in sorted({0, 1, 2, parameter})
        }
    if any(name in selected for name in ("marked", "realize", "lifting")):
        left = kernel_sequence(jump_tensor(2), modulus=modulus)
        right = kernel_sequence(jump_tensor(2) + 1, modulus=modulus)
    if "marked" in selected:
        records["marked"] = {
            "T": _summary(left, matrices), "T_plus_J": _summary(right, matrices),
            "same_kernel": _same_kernels(left, right),
        }
    if "spheres" in selected:
        records["spheres"] = {}
        for degree in range(1, max_degree + 1):
            tensor = SparseTensor(degree, (0,) * (degree + 1),
                                  {tuple(range(degree + 1)): 1})
            records["spheres"][str(degree)] = {
                "shape": tensor.shape, "stored_entries": tensor.nnz,
                **_summary(kernel_sequence(tensor, modulus=modulus), matrices),
            }
    if "realize" in selected:
        tensor = realize_kernel(left.inclusion, left.tensor.shifts)
        reconstructed = kernel_sequence(tensor, modulus=modulus)
        same = _same_kernels(left, reconstructed)
        if not same:
            raise AssertionError("The constructed tensor has a different kernel.")
        records["realize"] = {
            "same_kernel": same, "shape": tensor.shape,
            "entries": [[list(position), str(value)] for position, value in tensor.entries.items()],
            **_summary(reconstructed, matrices),
        }
    if "lifting" in selected:
        phi = chain_isomorphism(left.quotient, right.quotient)
        lifted = lift_quotient_isomorphism(left.projection, right.projection, phi)
        for degree in range(len(left.source.dimensions)):
            if field.matmul(right.projection.matrix(degree), lifted.matrix(degree)) != (
                    field.matmul(phi.matrix(degree), left.projection.matrix(degree))):
                raise AssertionError("The lift does not induce the specified quotient map.")
        records["lifting"] = {
            "invertible": lifted.is_isomorphism(),
            "ranks": [field.rank(value) for value in lifted.matrices],
        }
        if matrices:
            records["lifting"]["matrices"] = [_matrix(value) for value in lifted.matrices]
    if "obstruction" in selected:
        source = simplex_chain_complex(2, modulus=modulus)
        plane = field.nullspace(sp.Matrix([[1, 1, 1]]))
        kernel = ChainComplex(
            (2, 0, 0), (sp.zeros(0, 2), sp.zeros(2, 0), sp.zeros(0, 0)),
            modulus=modulus, lifts=(plane, sp.zeros(3, 0), sp.zeros(1, 0)),
            ambient_bases=source.ambient_bases,
        )
        inclusion = ChainMap(kernel, source, kernel.lifts)
        forbidden_plane = _capacity_error(lambda: realize_kernel(inclusion, (0, 0)))
        four_disks = ChainComplex((4, 4), (sp.zeros(0, 4), sp.eye(4)), modulus=modulus)
        boundary = sp.zeros(4, 5)
        for i in range(3):
            boundary[i, i] = 1
        upper = sp.zeros(5, 2)
        upper[3, 0] = upper[4, 1] = 1
        # This disk basis has the ranks of N(X((1,1);K)), including its H_0.
        positive_tails = ChainComplex((4, 5, 2),
                                      (sp.zeros(0, 4), boundary, upper), modulus=modulus)
        boundary_capacity = _capacity_error(lambda: chain_embedding(four_disks, positive_tails))
        records["obstruction"] = {"excluded_plane": forbidden_plane,
                                  "positive_tail_boundary": boundary_capacity}
    return {"coefficient_field": "Q" if modulus is None else f"GF({field.modulus})",
            "examples": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, help="Prime p for GF(p); default: Q")
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--max-degree", type=int, default=3,
                        help="Largest single-coordinate sphere degree; default: 3")
    parser.add_argument("--parameter", type=int, default=4,
                        help="Additional integer t in the matrix family; default: 4")
    parser.add_argument("--matrices", action="store_true", help="Include exact based matrices")
    args = parser.parse_args(argv)
    try:
        result = examples(modulus=args.modulus, case=args.case, max_degree=args.max_degree,
                          parameter=args.parameter, matrices=args.matrices)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
