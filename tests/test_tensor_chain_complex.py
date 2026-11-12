"""Exact chain representatives and sparse grading against independent summaries."""

from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np
import pytest
import sympy as sp
from sympy.polys.matrices import DomainMatrix

from simplicial_tensors.homology import generated_homology, relative_horn_homology
from simplicial_tensors.rank_shifted import SparseTensor, circle_generator
from simplicial_tensors.tensor_chain_complex import generated_chain_complex, sparse_generated_homology
from simplicial_tensors.tensor_ops import face


def _unnormalized_sparse_betti(tensor, modulus):
    """Independent dense pullbacks with explicitly bounded active indices."""
    dense = np.zeros(tensor.shape, dtype=object)
    for index, value in tensor.entries.items():
        dense[index] = value
    top = min(tensor.degree, len(tensor.shifts))
    tails = [tuple(range(tensor.degree + 1, tensor.degree + shift + 1))
             for shift in tensor.shifts]

    def rank(columns):
        if not columns:
            return 0
        data = []
        for column in columns:
            values = []
            for value in column.flat:
                rational = sp.Rational(value)
                values.append(rational if modulus is None else
                              int(rational.p) * pow(int(rational.q), -1, modulus) % modulus)
            data.append(values)
        domain = sp.QQ if modulus is None else sp.GF(modulus)
        return int(DomainMatrix.from_Matrix(sp.Matrix(data)).convert_to(domain).rank())

    dimensions, ranks = [], [0]
    for degree in range(top + 2):
        columns = [dense[np.ix_(*(vertices + tail for tail in tails))]
                   for vertices in combinations_with_replacement(range(tensor.degree + 1), degree + 1)]
        dimensions.append(rank(columns))
        if degree:
            ranks.append(rank([sum(((-1)**i * face(column, i) for i in range(degree + 1)),
                                   np.zeros_like(column[tuple(slice(0, -1) for _ in tensor.shifts)]))
                               for column in columns]))
    return tuple(dimensions[q] - ranks[q] - ranks[q + 1] for q in range(top + 1))


@pytest.mark.parametrize('modulus', [None, 2, 3])
@pytest.mark.parametrize('tensor', [
    np.array([1, 3, 8], dtype=object),
    np.array([[0, 1], [0, 0]], dtype=object),
    np.arange(12, dtype=object).reshape(3, 4),
    np.arange(8, dtype=object).reshape(2, 2, 2),
    np.array([[Fraction(1, 5), 2**60 + 1, 0]], dtype=object),
    np.zeros((3, 3), dtype=object),
])
def test_chain_models_match_existing_dense_summaries(tensor, modulus):
    model = generated_chain_complex(tensor, modulus=modulus)
    assert model.homology() == generated_homology(tensor, modulus=modulus)
    for q in range(len(model.dimensions)):
        cycles = model.cycles(q)
        representatives = model.homology_representatives(q)
        assert representatives.cols == model.homology().homology_dimensions[q]
        assert not any(model.field.matmul(model.differential(q), cycles))
        assert model.lift(q, representatives).rows == len(model.ambient_bases[q])
    if min(tensor.shape) > 1:
        for missing in range(min(tensor.shape)):
            quotient = generated_chain_complex(tensor, missing_face=missing, modulus=modulus)
            assert quotient.homology() == relative_horn_homology(tensor, missing, modulus=modulus)


def test_sparse_large_shifts_do_not_allocate_dense_tensors(monkeypatch):
    tensor = SparseTensor(1, (100, 100, 100), {(0, 1, 101): 1})

    def forbidden(*args, **kwargs):
        raise AssertionError('Dense ambient allocation is forbidden')

    monkeypatch.setattr(np, 'ndindex', forbidden)
    model = generated_chain_complex(tensor)
    assert tensor.shape == (102, 102, 102)
    assert model.dimensions == (0, 1)
    assert model.homology().homology_dimensions == (0, 1)
    assert len(model.ambient_bases[1]) == 1
    lifted = model.lift(1, model.homology_representatives(1))
    assert model.ambient_bases[1] == ((0, 1, 101),)
    assert lifted == sp.Matrix([[1]])


@pytest.mark.parametrize('modulus', [None, 2, 3])
def test_sparse_grading_and_modular_normalization(modulus):
    tensor = SparseTensor(2, (1, 2, 1), {
        (0, 1, 2): Fraction(2**53 + 1, 5),
        (3, 4, 0): Fraction(-7, 5),
        (1, 2, 1): Fraction(3, 5),
    })
    model = generated_chain_complex(tensor, modulus=modulus)
    assert model.homology().homology_dimensions == _unnormalized_sparse_betti(tensor, modulus)
    for q in range(1, len(model.dimensions)):
        assert not any(model.field.matmul(model.differential(q - 1), model.differential(q)))


def test_sparse_offsets_match_corresponding_dense_tensor():
    tensor = SparseTensor(2, (0, 1), {(0, 1): 2, (2, 3): 7})
    dense = np.zeros(tensor.shape, dtype=object)
    for index, value in tensor.entries.items():
        dense[index] = value
    assert sparse_generated_homology(tensor) == generated_homology(dense)


def test_chain_input_validation_and_degree_zero():
    assert generated_chain_complex(SparseTensor.zero(0, (2,))).dimensions == (0,)
    assert sparse_generated_homology(SparseTensor(0, (2,), {(1,): 7})).homology_dimensions == (1,)
    for tensor in [np.array(1), np.zeros((0, 2))]:
        with pytest.raises(ValueError, match='nonempty'):
            generated_chain_complex(tensor)
    with pytest.raises(TypeError, match='exact|Exact'):
        generated_chain_complex(np.array([1.0]))
    with pytest.raises(ValueError, match='denominator'):
        generated_chain_complex(SparseTensor(1, (0,), {(0,): Fraction(1, 2)}), modulus=2)
    with pytest.raises(TypeError, match='integer'):
        generated_chain_complex(circle_generator(), missing_face=True)
    with pytest.raises(ValueError, match='positive-degree'):
        generated_chain_complex(SparseTensor.zero(0, (0,)), missing_face=0)
    with pytest.raises(TypeError, match='SparseTensor'):
        sparse_generated_homology(np.zeros((2, 2)))
