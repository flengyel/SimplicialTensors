"""Shape/storage invariants shared by symbolic tensor backends."""

import numpy as np
import pytest
import sympy as sp

from simplicial_tensors.symbolic_tensor_ops import SymbolicTensor
from simplicial_tensors.sagemath_compatible_tensor_ops import SymbolicTensor as CompatibleTensor


@pytest.mark.parametrize("backend", [SymbolicTensor, CompatibleTensor])
@pytest.mark.parametrize("tensor", [[[1]], [1, 2, 3, 4], 1])
def test_constructor_rejects_data_with_a_different_shape(backend, tensor):
    with pytest.raises(ValueError, match="match the stated shape"):
        backend((2, 2), tensor=tensor)


@pytest.mark.parametrize("backend", [SymbolicTensor, CompatibleTensor])
@pytest.mark.parametrize("shape,error", [
    ((), ValueError), ((-1, 2), ValueError), ((True, 2), TypeError),
    ((1.0, 2), TypeError),
])
def test_constructor_checks_exact_integer_dimensions(backend, shape, error):
    with pytest.raises(error):
        backend(shape, tensor=[[1, 2]])


@pytest.mark.parametrize("backend", [SymbolicTensor, CompatibleTensor])
def test_constructor_normalizes_index_scalars_and_copies_storage(backend):
    data = np.array([[sp.Rational(2**60 + 1, 7), 3]], dtype=object)
    tensor = backend([np.int64(1), sp.Integer(2)], tensor=data)
    assert tensor.shape == (1, 2)
    assert all(type(size) is int for size in tensor.shape)
    data[0, 0] = 0
    assert tensor.tensor[0, 0] == sp.Rational(2**60 + 1, 7)


@pytest.mark.parametrize("backend", [SymbolicTensor, CompatibleTensor])
def test_empty_axes_from_degree_zero_faces_and_boundaries_remain_valid(backend):
    tensor = backend((1, 3), tensor=[[1, 2, 3]])
    assert tensor.face(0).shape == (0, 2)
    assert tensor.bdry().shape == (0, 2)
    empty = backend((0, 2), tensor=np.empty((0, 2), dtype=object))
    assert empty.tensor.shape == empty.shape
