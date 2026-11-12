"""Exact tensor homology against unnormalized chains and known complexes."""

import runpy
import subprocess
import sys
from fractions import Fraction
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from simplicial_tensors.homology import generated_homology, relative_horn_homology
from simplicial_tensors.tensor_ops import bdry, face

ROOT = Path(__file__).resolve().parents[1]


def column_rank(columns, field):
    if not columns:
        return 0
    entries = np.column_stack([column.ravel() for column in columns]).tolist()
    return DomainMatrix.from_Matrix(sp.Matrix(entries)).convert_to(field).rank()


def all_pullbacks(tensor, degree):
    active = min(tensor.shape)
    tails = [tuple(range(active, length)) for length in tensor.shape]
    return [tensor[np.ix_(*(vertices + tail for tail in tails))]
            for vertices in combinations_with_replacement(range(active), degree + 1)]


def unnormalized_betti(tensor, field, *, missing=None):
    """Compute the quotient complex before normalization, through degree n+1."""
    top_degree = min(tensor.shape) - 1
    included = [face(tensor, i) for i in range(top_degree + 1) if i != missing]
    if missing is None:
        included = []
    dimensions = []
    ranks = [0]
    previous_subspace = []
    previous_subrank = 0
    for degree in range(top_degree + 2):
        generators = all_pullbacks(tensor, degree)
        subspace = [column for facet in included for column in all_pullbacks(facet, degree)]
        subrank = column_rank(subspace, field)
        dimensions.append(column_rank(generators, field) - subrank)
        if degree:
            ranks.append(column_rank([bdry(column) for column in generators] + previous_subspace,
                                     field) - previous_subrank)
        previous_subspace, previous_subrank = subspace, subrank
    return tuple(dimensions[p] - ranks[p] - ranks[p + 1] for p in range(top_degree + 1))


@pytest.mark.parametrize("modulus,field", [(None, sp.QQ), (2, sp.GF(2)), (3, sp.GF(3))])
def test_normalization_agrees_with_all_order_preserving_maps(modulus, field):
    tensor = np.array([[1, 2, 3, 4], [5, 2, 1, 0], [0, 1, 1, 2]], dtype=object)
    expected = unnormalized_betti(tensor, field)
    assert generated_homology(tensor, modulus=modulus).homology_dimensions == expected
    for omitted in range(3):
        expected = unnormalized_betti(tensor, field, missing=omitted)
        assert relative_horn_homology(tensor, omitted, modulus=modulus).homology_dimensions == expected


def test_zero_vertex_circle_and_a_nontrivial_three_by_three_example():
    assert generated_homology(np.zeros((3, 3), dtype=object)).homology_dimensions == (0, 0, 0)
    assert generated_homology(np.array([[7]], dtype=object)).homology_dimensions == (1,)
    edge = np.array([[0, 1], [0, 0]], dtype=object)
    result = generated_homology(edge)
    assert result.chain_dimensions == (0, 1)
    assert result.boundary_ranks == (0, 0)
    assert result.homology_dimensions == (0, 1)
    tensor = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 2]], dtype=object)
    result = generated_homology(tensor)
    assert result.chain_dimensions == (1, 3, 1)
    assert result.boundary_ranks == (0, 1, 1)
    assert result.homology_dimensions == (0, 1, 0)


def test_horn_quotient_remembers_the_included_face_submodule():
    tensor = np.arange(8, dtype=object).reshape(2, 2, 2)
    assert relative_horn_homology(tensor, 0).homology_dimensions == (0, 0)
    assert relative_horn_homology(tensor, 1).homology_dimensions == (0, 1)


def test_exact_coefficients_and_field_validation():
    circle = np.array([[0, Fraction(1, 2)], [0, 0]], dtype=object)
    assert generated_homology(circle).homology_dimensions == (0, 1)
    assert generated_homology(circle, modulus=3).homology_dimensions == (0, 1)
    assert generated_homology(np.array([[0, 2**53 + 1], [0, 0]], dtype=object)).homology_dimensions == (0, 1)
    with pytest.raises(ValueError, match="denominator"):
        generated_homology(circle, modulus=2)
    with pytest.raises(TypeError, match="rational"):
        generated_homology(np.array([1.0]))
    for value in [0, 1, 4, 1.5, True]:
        with pytest.raises(ValueError, match="prime"):
            generated_homology(np.array([1]), modulus=value)
    for value in [np.array(1), np.array([]), np.zeros((0, 2))]:
        with pytest.raises(ValueError, match="nonempty"):
            generated_homology(value)
    with pytest.raises(TypeError, match="integer"):
        relative_horn_homology(np.ones((2, 2)), 0.0)
    with pytest.raises(ValueError, match="positive-degree"):
        relative_horn_homology(np.ones((1, 1), dtype=int), 0)


def test_experiment_keys_preserve_coordinates_and_degree_zero_rank():
    module = runpy.run_path(str(ROOT / "experiments/normalized_moore_complex_homology.py"))
    assert module["key_tensor"](np.array([[0, 1], [0, 0]])) != module["key_tensor"](np.array([[0, 0], [1, 0]]))
    symbolic = runpy.run_path(str(ROOT / "experiments/sagemath_normalized_tensor_homology.py"))
    homology_rank = symbolic["compute_homology_rank"]
    # Boundary of a path on three vertices: H_0 = Q.
    incidence = sp.Matrix([[-1, 0], [1, -1], [0, 1]])
    assert homology_rank({1: incidence}, 0) == 1
    assert homology_rank({1: incidence}, 1) == 0
    assert homology_rank({}, 0, chain_dimension=3) == 3
    assert homology_rank({0: sp.zeros(0, 3)}, 0) == 3
    with pytest.raises(ValueError, match="required"):
        homology_rank({}, 0)
    with pytest.raises(ValueError, match="compose"):
        homology_rank({1: sp.eye(2), 2: sp.eye(2)}, 1)
    with pytest.raises(ValueError, match="agree"):
        homology_rank({1: incidence}, 0, chain_dimension=2)
    with pytest.raises(TypeError, match="rational"):
        homology_rank({1: sp.Matrix([[0.5]])}, 1)


@pytest.mark.parametrize("script", ["homology_basis.py", "normalized_moore_complex_homology.py",
                                   "sagemath_normalized_tensor_homology.py"])
def test_homology_experiment_entry_points(script):
    completed = subprocess.run([sys.executable, str(ROOT / "experiments" / script)],
                               cwd=ROOT, capture_output=True, text=True, check=True, timeout=30)
    assert "Betti numbers" in completed.stdout


def test_legacy_notebook_uses_exact_homology_without_missing_dask_import():
    from simplicial_tensors.notebooks.homology_basis_with_degeneracy_check_dask import (
        compute_homology_dimension,
    )

    tensor = np.arange(8, dtype=object).reshape(2, 2, 2)
    assert compute_homology_dimension(tensor.shape, 0, tensor=tensor) == 0
    assert compute_homology_dimension(tensor.shape, 1, tensor=tensor) == 1


def test_second_legacy_homology_export_uses_actual_tensor_relations():
    from simplicial_tensors.notebooks.homology_basis_dask import (
        full_homology_verification,
    )

    zero = np.zeros((3, 3), dtype=object)
    assert full_homology_verification(zero.shape, tensor=zero) == 0
    circle = np.array([[0, 1], [0, 0]], dtype=object)
    assert full_homology_verification(circle.shape, tensor=circle) == 1


def test_cocalc_notebook_has_valid_json_and_python_cells():
    import ast
    import json

    path = ROOT / "src/simplicial_tensors/notebooks/cocalc_compatible_tensor_ops.ipynb"
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_cocalc_adapter_uses_shared_symbolic_operations():
    from simplicial_tensors.notebooks.cocalc_compatible_tensor_ops import (
        SymbolicTensor,
        correction_rank,
    )

    zero = SymbolicTensor((2, 2), init_type="zeros")
    assert zero.is_degen()
    assert not zero.n_hypergroupoid_comparison(outer_horns=True, allow_degen=True)
    assert zero.order() == 2
    edge = SymbolicTensor((2, 2), tensor_data=[[0, 1], [0, 0]])
    degenerate = edge.degen(0) + edge.degen(1)
    normalized, terms = degenerate.decompose_degen()
    assert degenerate.is_degen()
    assert not any(normalized.tensor.flat)
    reconstructed = normalized
    for lower, index in terms:
        reconstructed += lower.degen(index)
    np.testing.assert_array_equal(reconstructed.tensor, degenerate.tensor)
    x, y = sp.symbols("x y")
    differences = SymbolicTensor((2, 2), tensor_data=[[x, 2*x], [y, 3*y]])
    assert correction_rank(differences, zero) == 2
    assert "\\begin{bmatrix}" in edge.to_latex()
    assert edge.degen(0).order() == 2
    with pytest.raises(ValueError, match="shape"):
        SymbolicTensor((2, 2), tensor_data=[[1]])
