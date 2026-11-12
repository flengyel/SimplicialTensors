"""Exact field ranks, based complexes, representatives, and ambient lifts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from simplicial_tensors.chain_complex import ChainComplex, ExactField


@pytest.mark.parametrize("modulus,rank", [(None, 2), (2, 1), (3, 2)])
def test_rank_nullspace_and_coordinates_use_the_coefficient_field(modulus, rank):
    field = ExactField(modulus)
    matrix = field.matrix([[1, 2], [3, 4]])
    assert field.rank(matrix) == rank
    kernel = field.nullspace(matrix)
    assert kernel.shape == (2, 2 - rank)
    assert not any(field.matmul(matrix, kernel))
    basis = field.column_basis(matrix)
    coordinates = field.coordinates(basis, matrix)
    assert field.matmul(basis, coordinates) == matrix
    assert field.rank(basis) == basis.cols == rank


def test_exact_rationals_large_entries_and_rejected_coefficients():
    field = ExactField()
    large = 2**60 + 1
    assert field.matrix([[Fraction(large, 3)]])[0, 0] == sp.Rational(large, 3)
    assert ExactField(3).matrix([[Fraction(1, 2)]])[0, 0] == 2
    with pytest.raises(ValueError, match="denominator"):
        ExactField(2).matrix([[Fraction(1, 2)]])
    for value in [1.0, np.float64(0.5), sp.Float(0.5), sp.sqrt(2), "1/2"]:
        with pytest.raises(TypeError, match="integers or rationals"):
            field.matrix([[value]])
    for modulus in [0, 1, 4, -2, 2.0, True]:
        with pytest.raises(ValueError, match="prime"):
            ExactField(modulus)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_coordinates_reject_dependence_and_targets_outside_the_span(modulus):
    field = ExactField(modulus)
    with pytest.raises(ValueError, match="independent"):
        field.coordinates(sp.zeros(2, 1), sp.zeros(2, 1))
    with pytest.raises(ValueError, match="outside"):
        field.coordinates([[1], [0]], [[0], [1]])
    with pytest.raises(ValueError, match="same number"):
        field.coordinates(sp.eye(2), sp.zeros(3, 1))
    assert field.coordinates(sp.zeros(3, 0), sp.zeros(3, 2)).shape == (0, 2)
    assert field.nullspace(sp.zeros(0, 3)) == sp.eye(3)
    assert field.column_basis(sp.zeros(3, 0)).shape == (3, 0)


@pytest.mark.parametrize("modulus,betti", [(None, (1, 0, 0)), (2, (1, 1, 1)), (3, (1, 0, 0))])
def test_chain_representatives_and_field_dependent_homology(modulus, betti):
    model = ChainComplex((2, 2, 1), (sp.zeros(0, 2), sp.diag(1, 0), sp.Matrix([[0], [2]])),
                         modulus=modulus)
    assert model.homology().homology_dimensions == betti
    assert model.chain_dimensions == (2, 2, 1)
    for degree, expected in enumerate(betti):
        cycles = model.cycles(degree)
        boundaries = model.boundaries(degree)
        representatives = model.homology_representatives(degree)
        assert representatives.shape == (model.dimension(degree), expected)
        assert not any(model.field.matmul(model.differential(degree), cycles))
        assert not any(model.field.matmul(model.differential(degree), representatives))
        assert model.field.rank(boundaries.row_join(representatives)) == boundaries.cols + expected
        assert model.field.rank(cycles) == boundaries.cols + expected
        assert model.lift(degree, representatives) == representatives


def test_chain_validation_is_modular_and_snapshots_are_immutable():
    d1 = sp.Matrix([[2]])
    d2 = sp.Matrix([[1]])
    with pytest.raises(ValueError, match="compose to zero"):
        ChainComplex((1, 1, 1), (sp.zeros(0, 1), d1, d2))
    model = ChainComplex((1, 1, 1), (sp.zeros(0, 1), d1, d2), modulus=2)
    d1[0, 0] = 9
    assert model.differentials[1] == sp.zeros(1, 1)
    with pytest.raises(TypeError):
        model.differentials[1][0, 0] = 1
    with pytest.raises(FrozenInstanceError):
        model.dimensions = (9,)
    with pytest.raises(ValueError, match="expected"):
        ChainComplex((2,), (sp.zeros(1, 2),))
    with pytest.raises(ValueError, match="one differential"):
        ChainComplex((1,), ())
    with pytest.raises(ValueError, match="nonnegative"):
        ChainComplex((-1,), ())


def test_lifts_labels_and_empty_complex():
    lift = sp.Matrix([[1], [2], [3]])
    model = ChainComplex((1,), (sp.zeros(0, 1),), lifts=(lift,),
                         ambient_bases=(("a", "b", "c"),))
    lift[0, 0] = 9
    assert model.lift(0, [[2]]) == sp.Matrix([[2], [4], [6]])
    assert model.ambient_bases == (("a", "b", "c"),)
    with pytest.raises(ValueError, match="independent column"):
        ChainComplex((1,), (sp.zeros(0, 1),), lifts=(sp.zeros(2, 1),))
    with pytest.raises(ValueError, match="basis labels"):
        ChainComplex((1,), (sp.zeros(0, 1),), ambient_bases=(("a", "b"),))
    with pytest.raises(ValueError, match="one row"):
        model.lift(0, [[1], [2]])
    empty = ChainComplex((), ())
    assert empty.homology().chain_dimensions == ()
    assert empty.homology().homology_dimensions == ()
    assert empty.cycles(0).shape == empty.boundaries(0).shape == (0, 0)
    assert empty.homology_representatives(4).shape == (0, 0)
    assert empty.differential(3).shape == (0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        empty.cycles(-1)


@pytest.mark.parametrize("modulus,rank", [(None, 2), (2, 1), (3, 1)])
def test_sparse_rank_coerces_only_stored_entries(monkeypatch, modulus, rank):
    calls = []
    original = ExactField.scalar
    def record(field, value):
        calls.append(value)
        return original(field, value)
    monkeypatch.setattr(ExactField, "scalar", record)
    matrix = sp.ImmutableSparseMatrix(100, 150, {(0, 0): 2, (99, 149): 3})
    assert ExactField(modulus).rank(matrix) == rank
    assert len(calls) == matrix.nnz() == 2
