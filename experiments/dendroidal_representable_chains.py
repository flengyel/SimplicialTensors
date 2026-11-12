"""Compute normalized chains of two explicit dendroidal representables.

The fixtures are a binary corolla and a unary vertex above one input of a
binary vertex, together with their relative boundary quotients. Face signs
follow Basic--Nikolaus, Homology of dendroidal sets, Definition 3.1.
"""

from __future__ import annotations

import argparse
import json

import sympy as sp

from simplicial_tensors.chain_complex import ChainComplex, ExactField


EXPECTED = {
    "binary_corolla": ((3, 1), (0, 1), (2, 0)),
    "unary_binary_tree": ((4, 3, 1), (0, 2, 1), (2, 0, 0)),
    "boundary_binary_corolla": ((3,), (0,), (3,)),
    "boundary_unary_binary_tree": ((4, 3), (0, 2), (2, 1)),
    "relative_binary_corolla": ((0, 1), (0, 0), (0, 1)),
    "relative_unary_binary_tree": ((0, 0, 1), (0, 0, 0), (0, 0, 1)),
}


def representable_models(*, modulus: int | None = None) -> dict[str, ChainComplex]:
    """Return representables, their boundaries, and relative quotients."""
    field = ExactField(modulus)
    corolla_bases = (("r", "a", "b"), ("C2",))
    tree_bases = (("a", "e", "b", "r"), ("u", "v", "w"), ("T",))
    corolla = ChainComplex(
        (3, 1),
        (sp.zeros(0, 3), sp.Matrix([-1, 1, 1])),
        modulus=field.modulus,
        ambient_bases=corolla_bases,
    )
    # u:a->e, v:(e,b)->r, and their composite w:(a,b)->r.
    # Root pruning + top pruning - inner contraction gives u + v - w.
    tree = ChainComplex(
        (4, 3, 1),
        (sp.zeros(0, 4),
         sp.Matrix([[1, 0, 1], [-1, 1, 0], [0, 1, 1], [0, -1, -1]]),
         sp.Matrix([1, 1, -1])),
        modulus=field.modulus,
        ambient_bases=tree_bases,
    )
    boundary_corolla = ChainComplex(
        corolla.dimensions[:-1], corolla.differentials[:-1],
        modulus=field.modulus, ambient_bases=corolla_bases[:-1],
    )
    boundary_tree = ChainComplex(
        tree.dimensions[:-1], tree.differentials[:-1],
        modulus=field.modulus, ambient_bases=tree_bases[:-1],
    )
    relative_corolla = ChainComplex(
        (0, 1),
        (sp.zeros(0, 0), sp.zeros(0, 1)),
        modulus=field.modulus,
        lifts=(sp.zeros(3, 0), sp.eye(1)),
        ambient_bases=corolla_bases,
    )
    relative_tree = ChainComplex(
        (0, 0, 1),
        (sp.zeros(0, 0), sp.zeros(0, 0), sp.zeros(0, 1)),
        modulus=field.modulus,
        lifts=(sp.zeros(4, 0), sp.zeros(3, 0), sp.eye(1)),
        ambient_bases=tree_bases,
    )
    return {
        "binary_corolla": corolla,
        "unary_binary_tree": tree,
        "boundary_binary_corolla": boundary_corolla,
        "boundary_unary_binary_tree": boundary_tree,
        "relative_binary_corolla": relative_corolla,
        "relative_unary_binary_tree": relative_tree,
    }


def verify_models(*, modulus: int | None = None) -> dict:
    """Check the chain identity and known leaf and relative-cell ranks."""
    models = representable_models(modulus=modulus)
    summaries = {}
    for name, model in models.items():
        for degree in range(1, len(model.dimensions)):
            composite = model.field.matmul(model.differential(degree - 1),
                                           model.differential(degree))
            if any(composite):
                raise AssertionError(f"Consecutive differentials do not vanish: {name}")
        result = model.homology()
        actual = (result.chain_dimensions, result.boundary_ranks, result.homology_dimensions)
        if actual != EXPECTED[name]:
            raise AssertionError(f"Unexpected homology for {name}: {actual}")
        summaries[name] = {
            "chain_dimensions": result.chain_dimensions,
            "boundary_ranks": result.boundary_ranks,
            "betti_numbers": result.homology_dimensions,
        }
    return {"coefficient_field": "Q" if modulus is None else f"GF({int(modulus)})",
            "models": summaries}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, help="Prime p for GF(p); default: Q")
    args = parser.parse_args(argv)
    try:
        result = verify_models(modulus=args.modulus)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
