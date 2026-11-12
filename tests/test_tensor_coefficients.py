"""Regressions for coefficient arithmetic, normalization, and horn kernels."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp

from simplicial_tensors.tensor_ops import (
    bdry, cobdry, decompose_degen, decompose_degen_numpy, degen, face, filler, hbdry, horn,
    is_degen, n_hypergroupoid_comparison, random_tensor, split_normalized, vbdry,
)
from simplicial_tensors.symbolic_tensor_ops import (
    SymbolicTensor, is_generator_symbolic, tensor_filler_difference_rank,
)
from simplicial_tensors.sagemath_compatible_tensor_ops import (
    SymbolicTensor as CompatibleTensor, correction_rank,
)


def test_vector_faces_boundary_and_degeneracy():
    a = np.array([3, 5])
    np.testing.assert_array_equal(face(a, 0), [5])
    np.testing.assert_array_equal(face(a, 1), [3])
    np.testing.assert_array_equal(bdry(a), [2])
    np.testing.assert_array_equal(degen(a, 1), [3, 5, 5])
    assert bdry(np.array([3])).shape == (0,)
    np.testing.assert_array_equal(bdry(bdry(np.array([3, 5, 11, 17]))), [0, 0])


@pytest.mark.parametrize("dtype,value", [(np.int16, 30000), (np.int64, 2**62)])
def test_exact_integer_boundary_does_not_overflow(dtype, value):
    a = np.array([-value, value], dtype=dtype)
    result = bdry(a, exact=True)
    assert result.dtype == object
    assert result[0] == 2 * value
    assert type(result[0]) is int


def test_unsigned_exact_boundary_is_signed():
    np.testing.assert_array_equal(bdry(np.array([250, 1], dtype=np.uint8), exact=True), [-249])


def test_exact_matrix_boundaries_preserve_large_entries():
    large = 2**62
    a = np.array([[-large, large], [large, large]], dtype=np.int64)
    np.testing.assert_array_equal(hbdry(a, exact=True), [[2 * large, 0]])
    np.testing.assert_array_equal(vbdry(a, exact=True), [[2 * large], [0]])
    expected = degen(a.astype(object), 0) - degen(a.astype(object), 1)
    np.testing.assert_array_equal(cobdry(a, exact=True), expected)
    assert horn(a, 0).dtype == a.dtype


def test_random_tensor_supports_values_above_int16():
    a = random_tensor((2, 3), low=2**40, high=2**40 + 100, seed=21)
    assert a.dtype == np.int64
    assert np.all((a >= 2**40) & (a < 2**40 + 100))
    np.testing.assert_array_equal(a, random_tensor((2, 3), low=2**40, high=2**40 + 100, seed=21))


def test_boolean_boundary_and_filler_use_f2():
    a = np.array([[True, False, True], [False, True, True], [True, False, False]])
    expected = np.array([[True, False], [True, True]])
    np.testing.assert_array_equal(bdry(a), expected)
    np.testing.assert_array_equal(bdry(bdry(a)), [[False]])
    data = horn(a, 1)
    b = filler(data, 1)
    assert b.dtype == np.bool_
    np.testing.assert_array_equal(horn(b, 1), data)


def test_exact_degenerate_sum_is_not_a_single_degeneracy():
    a = np.array([[1, 1, 0], [1, 2, 1], [0, 1, 1]], dtype=object) * (2**80 + 1)
    assert all(not np.array_equal(a, degen(face(a, i), i)) for i in range(2))
    parts = decompose_degen(a)
    assert parts is not None
    np.testing.assert_array_equal(sum((degen(b, i) for i, b in enumerate(parts)), np.zeros_like(a)), a)
    assert all(not isinstance(value, sp.Float) for b in parts for value in b.flat)


def test_horn_and_exact_filler_preserve_large_integer_coefficients():
    a = np.arange(9, dtype=np.int64).reshape(3, 3) * (2**55) + 1
    data = horn(a, 1)
    np.testing.assert_array_equal(data[0], a[1:, 1:])
    b = filler(data, 1, exact=True)
    assert b.dtype == object
    np.testing.assert_array_equal(horn(b, 1), data)


@pytest.mark.parametrize("shape", [(4,), (3, 4), (3, 3, 4)])
def test_moore_projection_and_decomposition(shape):
    a = np.arange(np.prod(shape), dtype=object).reshape(shape)
    normal, parts = split_normalized(a)
    for i in range(1, min(shape)):
        assert not np.any(face(normal, i))
    reconstructed = normal + sum((degen(b, i) for i, b in enumerate(parts)), np.zeros_like(a))
    np.testing.assert_array_equal(reconstructed, a)


def test_degree_zero_degenerate_submodule_is_zero():
    assert decompose_degen(np.zeros((1, 3), dtype=int)) == []
    assert is_degen(np.zeros((1, 3), dtype=int))
    assert not is_degen(np.ones((1, 3), dtype=int))


def test_approximate_decomposition_preserves_complex_coefficients():
    lower = np.array([[1 + 2j, 3 - 4j], [2j, -1 + 1j]])
    a = degen(lower, 0)
    parts = decompose_degen_numpy(a)
    assert parts is not None
    np.testing.assert_allclose(sum((degen(b, i) for i, b in enumerate(parts)), np.zeros_like(a)), a)
    assert decompose_degen_numpy(np.zeros((1, 3))) == []


@pytest.mark.parametrize("backend", [SymbolicTensor, CompatibleTensor])
def test_symbolic_backends_test_the_sum_of_degeneracy_images(backend):
    x, y = sp.symbols("x y")
    a = np.array([[x, x, 0], [x, x + y, y], [0, y, y]], dtype=object)
    assert backend(a.shape, tensor=a).is_degen()
    assert not backend((2, 2), tensor=[[0, x], [0, 0]]).is_degen()


def test_symbolic_generator_excludes_a_sum_of_degeneracies():
    a = SymbolicTensor((3, 3), tensor=[[1, 1, 0], [1, 2, 1], [0, 1, 1]])
    assert not is_generator_symbolic(a)


@pytest.mark.parametrize("backend", [SymbolicTensor, CompatibleTensor])
def test_symbolic_selected_zero_filler_does_not_prove_uniqueness(backend):
    a = backend((3, 3), init_type="zeros")
    assert not a.n_hypergroupoid_comparison(allow_degen=True)
    assert backend((2, 2), init_type="zeros").n_hypergroupoid_comparison(allow_degen=True)


def test_selected_zero_filler_does_not_prove_uniqueness():
    assert not n_hypergroupoid_comparison(np.zeros((3, 3)), allow_degen=True)
    assert n_hypergroupoid_comparison(np.zeros((4, 4)), allow_degen=True)


@pytest.mark.parametrize("backend,rank", [(SymbolicTensor, tensor_filler_difference_rank),
                                        (CompatibleTensor, correction_rank)])
def test_correction_rank_detects_linear_dependence(backend, rank):
    x, y = sp.symbols("x y")
    a = backend((2, 2), tensor=[[x, 2 * x], [y, 3 * y]])
    zero = backend((2, 2), init_type="zeros")
    assert rank(a, zero) == 2
    assert rank(zero, zero) == 0
    assert rank(backend((2, 2), tensor=[[1, 2], [3, 4]]), zero) == 1
    with pytest.raises(ValueError, match="requires polynomials"):
        rank(backend((2, 2), tensor=[[sp.sin(x), 0], [0, 0]]), zero)


def test_import_does_not_reset_global_random_generators():
    script = """
import pickle
import random
import numpy as np
import sympy  # Initialize Sage/SymPy before measuring package-owned RNG effects.
random.seed(17)
np.random.seed(23)
before = pickle.dumps((random.getstate(), np.random.get_state()))
import simplicial_tensors.tensor_ops
after = pickle.dumps((random.getstate(), np.random.get_state()))
assert before == after
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


@pytest.mark.parametrize("backend,rank", [
    (SymbolicTensor, tensor_filler_difference_rank), (CompatibleTensor, correction_rank),
])
@pytest.mark.parametrize("constant", [sp.sqrt(2), sp.pi])
def test_correction_rank_rejects_irrational_constants(backend, rank, constant):
    original = backend((1,), tensor=[constant])
    zero = backend((1,), init_type="zeros")
    with pytest.raises(ValueError, match="requires polynomials"):
        rank(original, zero)
