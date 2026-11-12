#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    tensor_ops.py
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


import sympy as sp
import numpy as np
from typing import Tuple, List, Union, Any
from . import tensor_ops as ops
from .tensor_ops import (n_hypergroupoid_conjecture, 
                       SimplicialException)  # Import the exception



class SymbolicTensor:

    def __init__(self, shape: Tuple[int, ...], tensor=None, init_type: str = 'range'):
        """Initialize a symbolic tensor."""
        self.shape = shape

        if tensor is not None:
            self.tensor = np.array(tensor, dtype=object, copy=True)
        else:
            self.tensor = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                idx_str = ','.join(map(str, idx))
                if init_type == 'range':
                    self.tensor[idx] = sp.Symbol(f'x_{{{idx_str}}}')
                elif init_type == 'zeros':
                    self.tensor[idx] = sp.S.Zero
                elif init_type == 'ones':
                    self.tensor[idx] = sp.S.One
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

    # Tensor addition 
    def __add__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Cannot add tensors of different shapes {self.shape} vs {other.shape}")
        
        sum_tensor = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            sum_tensor[idx] = self.tensor[idx] + other.tensor[idx]
        return SymbolicTensor(self.shape, tensor=sum_tensor)
    
    def __sub__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Cannot subtract tensors of different shapes {self.shape} vs {other.shape}")
        # elementwise
        diff = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            diff[idx] = self.tensor[idx] - other.tensor[idx]
        return SymbolicTensor(self.shape, tensor=diff)
    
    @staticmethod
    def from_tensor(tensor):
        """Create a SymbolicTensor from an existing tensor"""
        shape = tensor.shape
        return SymbolicTensor(shape, tensor=tensor)

    def dimen(self) -> int:
        return min(self.shape) - 1

    def order(self) -> int:
        """
        Calculates the order of this tensor, defined as the number of its dimensions.
        """
        return len(self.shape)

    def _dims(self) -> Tuple[np.ndarray, ...]:
        """
        Get dimensions of the tensor as a tuple of arrays of indices.
        This matches tensor_ops._dims()
        """
        return tuple([np.arange(start=0, stop=dim_size) for dim_size in self.shape])
    
    def face(self, i: int) -> "SymbolicTensor":
        """Delete the common index i in every axis."""
        return self.from_tensor(ops.face(self.tensor, i))


    def degen(self, k: int) -> "SymbolicTensor":
        """Repeat the common index k in every axis."""
        return self.from_tensor(ops.degen(self.tensor, k))


    def bdry(self) -> "SymbolicTensor":
        """Compute the alternating diagonal boundary over the scalar domain."""
        return self.from_tensor(ops.bdry(np.asarray(self.tensor, dtype=object)))

    
    def horn(self, k: int) -> List["SymbolicTensor"]:
        """Compute all faces with the omitted face represented by a zero tensor."""
        return [self.from_tensor(value) for value in ops.horn(self.tensor, k)]

    
    def filler(self, horn_list: List["SymbolicTensor"], k: int) -> "SymbolicTensor":
        """Use Moore's algorithm on a compatible horn."""
        return self.from_tensor(ops.filler(np.stack([face.tensor for face in horn_list]), k))

    
    def n_hypergroupoid_comparison(self, outer_horns=False, verbose=False, allow_degen=False) -> bool:
        """
        Test the n-hypergroupoid conjecture: uniqueness of fillers.
        Matches tensor_ops.n_hypergroupoid_comparison().
        """
        dim = self.dimen()
        horn_range = range(0 if outer_horns else 1, dim + 1 if outer_horns else dim)
        if dim < 1 or not horn_range:
            return True
        boundary = self.bdry()
        if not allow_degen and boundary.is_degen():
            if verbose:
                print("Boundary is degenerate.")
            raise SimplicialException("Degenerate boundary.")

        dim = self.dimen()  # Updated to use self.dimen() instead of dimen(self.tensor)
        horn_range = range(0 if outer_horns else 1, dim + 1 if outer_horns else dim)

        for i in horn_range:
            if verbose:
                print(f"Testing horn {i}...")

            horn_i = self.horn(i)
            filler_i = self.filler(horn_i, i)
            horn_i_prime = filler_i.horn(i)

            for j in range(len(horn_i)):
                if j == i:
                    continue
                original = horn_i[j]
                reproduced = horn_i_prime[j]
                for idx in np.ndindex(original.shape):
                    diff = sp.simplify(original.tensor[idx] - reproduced.tensor[idx])
                    if diff != 0:
                        if verbose:
                            print(f"Disagreement at face {j}, index {idx}: {diff}")
                        raise SimplicialException(f"Original horn and filler horn disagree at face {j}, position {idx}.")

            monomial_count = lambda expr: len(expr.as_ordered_terms()) if expr != 0 else 0
            differences = []
            indices_with_correction_terms = 0
            for idx in np.ndindex(self.shape):
                orig = self.tensor[idx]
                fill = filler_i.tensor[idx]
                diff = sp.simplify(orig - fill)
                if diff != 0:
                    count = monomial_count(fill)
                    differences.append((idx, orig, fill, count))
                    indices_with_correction_terms += 1

            if differences:
                if verbose:
                    print("Multiple fillers exist. The original tensor and the filler differ at the following indices:")
                    for idx, orig, fill, count in differences:
                        print(f"  At index {idx}:")
                        print(f"    Original: {sp.pretty(orig)}")
                        print(f"    Filler:   {sp.pretty(fill)}")
                        print(f"    Monomial count: {count}")
                    print(f"    Indices with correction terms: {indices_with_correction_terms}")
                return False

        unique = n_hypergroupoid_conjecture(self.shape)
        if verbose:
            print("Unique filler." if unique else "Nonzero horn kernel. Multiple fillers exist.")
        return unique

    ############### Simplification and Substitution Methods ###############    
    def simplify(self) -> "SymbolicTensor":
        """Simplify all expressions in the tensor"""
        for idx in np.ndindex(self.shape):
            self.tensor[idx] = sp.simplify(self.tensor[idx])
        return self
    
    def subs(self, substitutions: dict) -> "SymbolicTensor":
        """Apply substitutions to all expressions"""
        for idx in np.ndindex(self.shape):
            val = self.tensor[idx]
            subs_method = getattr(val, "subs", None)
            if callable(subs_method):
                # Only call subs if the element provides such a method (e.g., a SymPy expression)
                self.tensor[idx] = subs_method(substitutions)
            else:
                # Leave non-symbolic entries unchanged
                self.tensor[idx] = val
        return self
    
    def to_latex(self) -> str:
        """Convert to LaTeX representation"""
        if len(self.shape) == 2:
            rows, cols = self.shape
            latex = "\\begin{bmatrix}\n"
            for i in range(rows):
                for j in range(cols):
                    latex += sp.latex(self.tensor[i, j])
                    if j < cols - 1:
                        latex += " & "
                if i < rows - 1:
                    latex += " \\\\\n"
            latex += "\n\\end{bmatrix}"
            return latex
        else:
            # For higher dimensions, return a list of 2D slices
            slices = []
            if len(self.shape) == 3:
                for i in range(self.shape[0]):
                    slice_tensor = SymbolicTensor((self.shape[1], self.shape[2]), 
                                                 tensor=self.tensor[i, :, :])
                    slices.append(f"Slice {i}:\n{slice_tensor.to_latex()}")
            return "\n\n".join(slices)
    
    def __str__(self) -> str:
        """String representation of the tensor"""
        return str(self.tensor)
    
    def __repr__(self) -> str:
        """Representative string of the tensor"""
        return f"SymbolicTensor(shape={self.shape})"


    def decompose_degen(self) -> Union[List["SymbolicTensor"], None]:
        """Decompose a sum of degeneracies using the Moore projection."""
        if self.dimen() < 0:
            return None
        normalized, terms = ops.split_normalized(self.tensor)
        if any(sp.simplify(value) != 0 for value in normalized.flat):
            return None
        return [self.from_tensor(term) for term in terms]

    
    def is_degen(self) -> bool:
        """
        Determine if the symbolic tensor is degenerate by attempting to find
        an exact additive decomposition.
        """
        return self.decompose_degen() is not None

def _polynomial_coefficient_rank(expressions) -> int:
    """Rank the coefficient vectors of polynomial expressions over Q.

    Columns correspond to monomials, including the constant monomial.
    Nonpolynomial expressions raise ValueError.
    """
    expressions = [sp.expand(sp.sympify(expr)) for expr in expressions]
    variables = sorted(set().union(*(expr.free_symbols for expr in expressions)), key=str)
    if not variables:
        return int(any(expr != 0 for expr in expressions))
    try:
        polynomials = [sp.Poly(expr, *variables, domain=sp.QQ) for expr in expressions]
    except (sp.PolynomialError, sp.polys.polyerrors.CoercionFailed) as exc:
        raise ValueError("Correction rank requires polynomials with rational coefficients.") from exc
    monomials = sorted(set().union(*(poly.monoms() for poly in polynomials)))
    coefficients = sp.Matrix([[poly.coeff_monomial(monomial) for monomial in monomials]
                              for poly in polynomials])
    return int(coefficients.rank())


def tensor_filler_difference_rank(original: "SymbolicTensor", filler: "SymbolicTensor") -> int:
    """Rank the polynomial coefficient matrix of the entry differences.

    For homogeneous linear entries in independent symbols this is the rank
    of the represented correction map. Scalar multiples count once.
    """
    if original.shape != filler.shape:
        raise ValueError("Tensors must have the same shape to compare.")
    return _polynomial_coefficient_rank((original.tensor - filler.tensor).flat)


def test_symbolic_n_hypergroupoid(shape: Tuple[int, ...], verbose=True):
    """
    Test the n-hypergroupoid conjecture using symbolic tensors.

    Args:
        shape: Shape of the tensor to test
        verbose: Whether to print verbose output

    Returns:
        Tuple of (conjecture_result, comparison_result, symbolic_tensor)
    """
    # Create a symbolic range tensor
    sym_tensor = SymbolicTensor(shape)

    # Evaluate the conjecture condition: should uniqueness hold?
    conjecture = n_hypergroupoid_conjecture(shape, verbose=verbose)

    try:
        # Compare symbolic horn fillers
        comparison = sym_tensor.n_hypergroupoid_comparison(outer_horns=True, verbose=verbose)

        if verbose:
            print(f"Horn-kernel criterion predicts unique fillers: {conjecture}")
            print(f"Filler comparison and kernel criterion: {comparison}")
            if conjecture == comparison:
                print("Moore filler checks agree with the horn-kernel criterion.")
            else:
                print("Moore filler checks disagree with the horn-kernel criterion.")

        return conjecture, comparison, sym_tensor

    except SimplicialException as e:
        if "Degenerate boundary" in str(e):
            if verbose:
                print("Skipping comparison due to degenerate boundary.")
            return conjecture, None, sym_tensor
        raise

def is_generator_symbolic(T: SymbolicTensor) -> bool:
    """Test that T is nonzero and outside the additive degeneracy submodule."""
    return any(sp.simplify(value) != 0 for value in T.tensor.flat) and not T.is_degen()


def check_missing_symbols(t, t_prime, horn_faces, k):
    """
    For symbolic tensors t and t_prime (both SymbolicTensor or at least
    numpy arrays of sympy expressions), ensures t_prime differs from t exactly
    at the 'missing' symbols. Missing symbols are those that do not appear
    in any non-missing face of horn_faces.

    t: original SymbolicTensor
    t_prime: filler result (also SymbolicTensor)
    horn_faces: result of t.horn(k)
    k: the face index that is replaced by zeros in horn_faces

    Returns True if everything matches, else False.
    """
    shape = t.shape
    n = t.dimen()  # simplicial dimension of t

    print(f"Checking horn({n},{k}) indices missing from symbolic tensor with shape {shape}.")
    # Gather set of all symbols in T (by name).
    # For example, each T.tensor[idx] is x_{0,1} etc. If T might have zeros, skip them.
    all_symbols = set()
    for idx in np.ndindex(shape):
        expr = t.tensor[idx]
        if expr != sp.S.Zero:
            all_symbols.add(str(expr))

    # Gather union of symbol names in the non-missing faces
    face_symbol_union = set()
    for face_idx, face in enumerate(horn_faces):
        if face_idx == k:  # skip missing (zero) face
            continue
        fshape = face.shape
        for subidx in np.ndindex(fshape):
            expr = face.tensor[subidx]
            # If it isn't literally zero, gather its name
            if sp.simplify(expr) != sp.S.Zero:
                face_symbol_union.add(str(expr))

    # Missing = all symbols not in face_symbol_union
    missing_symbols = all_symbols - face_symbol_union

    # Now we see which entries Tprime changed vs. T.
    # We'll gather the symbolic name of T[idx] or Tprime[idx].
    changed_symbols = set()
    for idx in np.ndindex(shape):
        expr_orig = t.tensor[idx]
        expr_new = t_prime.tensor[idx]
        diff_expr = sp.simplify(expr_new - expr_orig)
        if diff_expr != sp.S.Zero:
            # We say that t'[idx] differs from t[idx]. So we record the name of the original symbol.
            # If the original was zero, there's no symbol to record, so we might record str(expr_new).
            if expr_orig == sp.S.Zero:
                changed_symbols.add(str(expr_new))
            else:
                changed_symbols.add(str(expr_orig))

    if changed_symbols == missing_symbols:
        print(f"Success: the filler differed from the original at {len(missing_symbols)} indices.")
        return True
    else:
        print("Mismatch in correction terms vs. missing symbols.")
        extra = changed_symbols - missing_symbols
        missed = missing_symbols - changed_symbols
        if extra:
            print("Symbols changed that were not missing:", extra)
        if missed:
            print("Symbols missing but unchanged:", missed)
        return False


def main():
    # Example usage
    shape = (3, 3)  # A shape where fillers are not unique
    # Test the conjecture
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)
    
    # Examine the specific case of a 1-horn
    horn_1 = sym_tensor.horn(1)
    filler_1 = sym_tensor.filler(horn_1, 1)
    
    # Compare the original tensor and its filler
    print("Original tensor:")
    print(sym_tensor)
    print("\nFiller tensor:")
    print(filler_1)
    
    print("\nComparison of original and filler tensors:")
    result = check_missing_symbols(sym_tensor, filler_1, horn_1, 1)
    print("Check result:", result)
    
    
    def build_shape(n: int) -> Tuple[int, ...]:
        return tuple(n+1 for _ in range(n))
    
    for k in range(3, 6):
        for j in range(1,k):
            shape = build_shape(k)
            print(f"Building inner Horn({k},{j}) of generic tensor of shape: {shape}")
            sym_tensor = SymbolicTensor(shape=shape)
            inner_horn = sym_tensor.horn(j)
            filler = sym_tensor.filler(inner_horn, j)
            result = check_missing_symbols(sym_tensor, filler, inner_horn, j)
            print(f"Result for shape {shape}: {result}")
    
    
#    for d in range(2, 7):
#        print(f"build_shape({d}): {build_shape(d)}")


#    for d in range(2, 7):
#        shape = build_shape(d)
#        conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)

    shape = (3,3,3)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)
    print(f"Shape: {shape}, Conjecture: {conjecture}, Comparison: {comparison}")
