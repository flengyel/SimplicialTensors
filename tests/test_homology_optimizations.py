"""Exact field ranks and normalization reuse for tensor homology."""

from fractions import Fraction

import numpy as np
import pytest
import sympy as sp
from sympy.polys.matrices import DomainMatrix

import simplicial_tensors.homology as homology
from simplicial_tensors.tensor_ops import split_normalized


@pytest.mark.parametrize("field", [sp.QQ, sp.GF(2), sp.GF(3)])
def test_direct_domain_rank_matches_dense_exact_reference(field):
    large = 2**70 + 1
    columns = [
        np.array([Fraction(large, 5), Fraction(1, 7), 0], dtype=object),
        np.array([Fraction(large, 5), Fraction(1, 7), 0], dtype=object),
        np.array([0, 3, 6], dtype=object),
        np.zeros(3, dtype=object),
        np.array([3, 6, 9], dtype=object),
    ]
    modulus = int(field.characteristic())
    if modulus:
        columns = [np.array([
            Fraction(value).numerator * pow(Fraction(value).denominator, -1, modulus) % modulus
            for value in column
        ], dtype=object) for column in columns]
    reference = DomainMatrix.from_Matrix(
        sp.Matrix(np.column_stack(columns).tolist())
    ).convert_to(field).rank()
    assert homology._column_rank(columns, field) == reference
    assert homology._column_rank([np.zeros(3, dtype=object)], field) == 0
    assert homology._column_rank([], field) == 0


def test_repeated_restrictions_reuse_normalization(monkeypatch):
    tensor = np.zeros((5, 7), dtype=object)
    calls = []
    original = homology.split_normalized

    def record_normalization(array):
        calls.append(array.shape)
        return original(array)

    monkeypatch.setattr(homology, "split_normalized", record_normalization)
    result = homology.relative_horn_homology(tensor, 0)
    assert result.homology_dimensions == (0, 0, 0)
    assert calls == [(1, 3), (2, 4), (3, 5)]


def test_normalization_cache_preserves_shape_and_exact_input_values():
    cache = {}
    first = np.array([[Fraction(2**60 + 1, 7), 2, 3], [4, 5, 6]], dtype=object)
    second = np.array([[Fraction(2**60 + 2, 7), 2, 3], [4, 5, 6]], dtype=object)
    actual_first = homology._normalized_columns(first, 1, cache)[0]
    actual_second = homology._normalized_columns(second, 1, cache)[0]
    np.testing.assert_array_equal(actual_first, split_normalized(first)[0])
    np.testing.assert_array_equal(actual_second, split_normalized(second)[0])
    assert not np.array_equal(actual_first, actual_second)
    assert homology._normalized_columns(np.array([1, 2], dtype=object), 1, cache)[0].shape == (2,)
