#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    test_sagemath_compatible_tensor_ops.py
#    
#    Copyright (C) 2021-2025 Florian Lengyel
#    Email: florian.lengyel at cuny edu, florian.lengyel at gmail
#    Website: https://github.com/flengyel
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

# This is a pytest suite adapted to test the SageMath-compatible version of SymbolicTensor.
# It should be run within a SageMath environment.

import pytest
import random

np = pytest.importorskip("numpy")
sage_all = pytest.importorskip("sage.all")
sage_symbolic = pytest.importorskip("sage.symbolic.expression")
var = sage_all.var
simplify = sage_all.simplify
Expression = sage_symbolic.Expression

# Import the class and helpers from the Sage-compatible file
# Ensure sagemath_compatible_tensor_ops.py is in the same directory or Python path.
from simplicial_tensors.sagemath_compatible_tensor_ops import (
    HAVE_SAGE, SymbolicTensor, SimplicialException, correction_rank,
    n_hypergroupoid_conjecture,
)

# Seed for reproducibility
___SEED___ = 42
random.seed(___SEED___)

def test_multiindex_variable_names():
    """Tests if variables are created with the correct Sage-compatible names."""
    shape = (2, 3)
    tensor = SymbolicTensor(shape)
    expected = np.array([
        [var("x_0_0"), var("x_0_1"), var("x_0_2")],
        [var("x_1_0"), var("x_1_1"), var("x_1_2")]
    ], dtype=object)
    assert tensor.shape == expected.shape
    for idx in np.ndindex(shape):
        assert tensor.tensor[idx] == expected[idx]

def test_face_degen_preserve_structure():
    """Tests if face and degen operations produce tensors of the correct shape."""
    shape = (3, 3)
    T = SymbolicTensor(shape)
    face = T.face(1)
    assert face.shape == (2, 2)
    degen = face.degen(1)
    assert degen.shape == shape

def test_boundary_simple():
    """Tests the boundary of a simple tensor."""
    T = SymbolicTensor((3, 3))
    bdry = T.bdry()
    assert bdry.shape == (2, 2)
    # Check if the result is a Sage symbolic expression
    assert isinstance(bdry.tensor[0, 0], Expression)

def test_filler_agrees_with_horn():
    """Tests if the filler correctly reconstructs the faces of the horn."""
    shape = (3, 3)
    T = SymbolicTensor(shape)
    horn_list = T.horn(1)
    filler = T.filler(horn_list, 1)
    
    # After filling, the faces of the filler should match the original horn faces
    for j, original_face in enumerate(horn_list):
        if j == 1: continue # Skip the missing face
        
        filler_face = filler.face(j)
        diff = original_face - filler_face
        
        # All entries in the difference tensor should simplify to zero
        assert np.all([(simplify(d) == 0) for d in diff.tensor.flatten()])

def test_bdry_squared_zero():
    """Tests the fundamental simplicial identity d^2 = 0."""
    for _ in range(5):
        dims = random.randint(2, 3)
        shape = tuple(random.randint(3, 5) for _ in range(dims))
        T = SymbolicTensor(shape)
        bdry1 = T.bdry()
        bdry2 = bdry1.bdry()
        # All entries in the second boundary should be zero
        assert all(simplify(entry) == 0 for entry in bdry2.tensor.flatten())

def test_simplicial_identities():
    """
    Groups all five simplicial identities into one parameterized test.
    This structure makes it easy to see which identity might fail.
    """
    
    # d_i d_j = d_{j-1} d_i for i < j
    def first_identity(T):
        n = T.dimen()
        for j in range(n + 1):
            for i in range(j):
                lhs = T.face(j).face(i)
                rhs = T.face(i).face(j - 1)
                assert np.all([(simplify(l-r)==0) for l,r in zip(lhs.tensor.flatten(), rhs.tensor.flatten())])
    
    # s_i s_j = s_{j+1} s_i for i <= j
    def fifth_identity(T): # Renamed from original test file for logical order
        n = T.dimen()
        for j in range(n + 1):
            for i in range(j + 1):
                lhs = T.degen(j).degen(i)
                rhs = T.degen(i).degen(j + 1)
                assert np.all([(simplify(l-r)==0) for l,r in zip(lhs.tensor.flatten(), rhs.tensor.flatten())])

    # d_i s_j relations
    def mixed_identities(T):
        n = T.dimen()
        for j in range(n + 1): # s_j
            sjT = T.degen(j)
            for i in range(n + 2): # d_i
                lhs = sjT.face(i)
                if i < j:
                    rhs = T.face(i).degen(j - 1)
                elif i == j or i == j + 1:
                    rhs = T # Identity
                else: # i > j + 1
                    rhs = T.face(i - 1).degen(j)
                
                assert np.all([(simplify(l-r)==0) for l,r in zip(lhs.tensor.flatten(), rhs.tensor.flatten())])

    for _ in range(3): # Run a few random trials
        dims = random.randint(2, 3)
        shape = tuple(random.randint(4, 5) for _ in range(dims))
        T = SymbolicTensor(shape)
        
        first_identity(T)
        fifth_identity(T)
        mixed_identities(T)

def test_n_hypergroupoid_conjecture_uniqueness():
    """Tests the conjecture's prediction against filler observation."""
    
    # Case 1: k >= N, expect non-unique
    shape_non_unique = (3, 3) # k=2, N=2
    T_non_unique = SymbolicTensor(shape_non_unique)
    conjecture_pred_1 = n_hypergroupoid_conjecture(shape_non_unique)
    diff = T_non_unique - T_non_unique.filler(T_non_unique.horn(1), 1)
    observed_unique_1 = not np.any(diff.simplify().tensor != 0)
    assert conjecture_pred_1 == False
    assert observed_unique_1 == False

    # Case 2: k < N, expect unique
    shape_unique = (5, 5) # k=2, N=4
    T_unique = SymbolicTensor(shape_unique)
    conjecture_pred_2 = n_hypergroupoid_conjecture(shape_unique)
    diff = T_unique - T_unique.filler(T_unique.horn(1), 1)
    observed_unique_2 = not np.any(diff.simplify().tensor != 0)
    assert conjecture_pred_2 == True
    assert observed_unique_2 == True


def test_native_sage_backend_is_active():
    assert HAVE_SAGE
    assert isinstance(SymbolicTensor((2, 2)).tensor[0, 0], Expression)


@pytest.mark.parametrize("i,j", [(0, 2), (2, 0)])
def test_mixed_identity_exact_matrix(i, j):
    tensor = SymbolicTensor((3, 3), tensor=np.arange(9, dtype=object).reshape(3, 3))
    lhs = tensor.degen(j).face(i)
    rhs = tensor.face(i).degen(j - 1) if i < j else tensor.face(i - 1).degen(j)
    np.testing.assert_array_equal(lhs.tensor, rhs.tensor)


def test_native_sage_correction_rank():
    x, y = var("sage_rank_x sage_rank_y")
    QQ = sage_all.QQ
    zero = SymbolicTensor((2, 2), init_type="zeros")
    original = SymbolicTensor((2, 2), tensor=[[x, 2*x], [y, 3*y]])
    rational = SymbolicTensor((2, 2), tensor=[
        [QQ(1)/3*x, QQ(2)/3*x], [QQ(5)/7*y, QQ(15)/7*y],
    ])
    assert correction_rank(original, zero) == 2
    assert correction_rank(rational, zero) == 2
    assert correction_rank(zero, zero) == 0
    assert correction_rank(SymbolicTensor((2, 2), tensor=[
        [QQ(2**60 + 1)/3, QQ(1)/7], [0, 0],
    ]), zero) == 1


@pytest.mark.parametrize("expression", [
    sage_all.sin(var("sage_nonpoly_x")),
    sage_all.sqrt(2)*var("sage_nonpoly_x"),
    sage_all.sqrt(2),
])
def test_native_sage_correction_rank_rejects_nonrational_polynomials(expression):
    tensor = SymbolicTensor((1,), tensor=[expression])
    with pytest.raises(ValueError, match="requires polynomials"):
        correction_rank(tensor, SymbolicTensor((1,), init_type="zeros"))


def test_native_sage_sum_of_degeneracies():
    x, y = var("sage_degen_x sage_degen_y")
    first = SymbolicTensor((2, 2), tensor=[[x, 0], [0, 0]]).degen(0)
    second = SymbolicTensor((2, 2), tensor=[[0, 0], [0, y]]).degen(1)
    tensor = first + second
    assert tensor.is_degen()
    for index in range(2):
        difference = tensor - tensor.face(index).degen(index)
        assert any(bool(simplify(value) != 0) for value in difference.tensor.flat)


@pytest.mark.parametrize("shape", [(1,), (1, 3)])
def test_native_sage_degree_zero_degeneracy(shape):
    assert SymbolicTensor(shape, init_type="zeros").is_degen()
    tensor = SymbolicTensor(shape, tensor=np.full(shape, var("sage_degree_zero"), dtype=object))
    assert not tensor.is_degen()


@pytest.mark.parametrize("shape,unique", [((3, 3), False), ((4, 4), True)])
@pytest.mark.parametrize("outer_horns", [False, True])
def test_native_sage_zero_filler_uniqueness(shape, unique, outer_horns):
    zero = SymbolicTensor(shape, init_type="zeros")
    assert zero.n_hypergroupoid_comparison(
        allow_degen=True, outer_horns=outer_horns,
    ) is unique


def _exact_sympy_array(array):
    import sympy as sp

    result = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        rational = sage_all.QQ(array[index])
        result[index] = sp.Rational(int(rational.numerator()), int(rational.denominator()))
    return result


@pytest.mark.parametrize("shape", [(4,), (3, 4), (3, 3, 4)])
def test_native_sage_exact_operations_agree_with_sympy(shape):
    from simplicial_tensors.symbolic_tensor_ops import SymbolicTensor as SympyTensor

    QQ = sage_all.QQ
    values = np.array([
        QQ(2**60 + 1)/3 + QQ(index)/7 for index in range(int(np.prod(shape)))
    ], dtype=object).reshape(shape)
    tensor = SymbolicTensor(shape, tensor=values)
    reference = SympyTensor(shape, tensor=_exact_sympy_array(values))

    def compare(actual, expected):
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(_exact_sympy_array(actual.tensor), expected.tensor)

    compare(tensor + tensor, reference + reference)
    compare(tensor - tensor, reference - reference)
    compare(tensor.bdry(), reference.bdry())
    compare(tensor.bdry().bdry(), reference.bdry().bdry())
    for index in range(tensor.dimen() + 1):
        compare(tensor.face(index), reference.face(index))
        compare(tensor.degen(index), reference.degen(index))
        horn = tensor.horn(index)
        reference_horn = reference.horn(index)
        for actual_face, expected_face in zip(horn, reference_horn):
            compare(actual_face, expected_face)
        actual_filler = tensor.filler(horn, index)
        compare(actual_filler, reference.filler(reference_horn, index))
        for actual_face, expected_face in zip(actual_filler.horn(index), horn):
            compare(actual_face, expected_face)



def test_native_sage_integer_shape_and_empty_axes():
    tensor = SymbolicTensor(
        (sage_all.ZZ(1), sage_all.ZZ(2)), tensor=[[sage_all.QQ(1)/7, 3]],
    )
    assert tensor.shape == (1, 2)
    assert tensor.face(0).shape == (0, 1)
    with pytest.raises(ValueError, match="match the stated shape"):
        SymbolicTensor((sage_all.ZZ(2), sage_all.ZZ(2)), tensor=[[1]])


if __name__ == "__main__":
    # To run these tests, execute from a SageMath command line:
    # sage -python -m pytest test_sagemath_compatible_tensor_ops.py
    pytest.main([__file__])
