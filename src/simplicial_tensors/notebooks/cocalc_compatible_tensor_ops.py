#    Opérations Tensorielles Simpliciales
#    Simplicial Operations on Matrices and Hypermatrices
#    sagemath_compatible_tensor_ops.py
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


"""CoCalc notebook adapter for the shared symbolic tensor implementation."""

import numpy as np

from ..sagemath_compatible_tensor_ops import (
    HAVE_SAGE,
    SimplicialException,
    check_symbolic_corrections,
    correction_rank,
    n_hypergroupoid_conjecture,
    test_symbolic_n_hypergroupoid,
)
from ..sagemath_compatible_tensor_ops import (
    SymbolicTensor as _SymbolicTensor,
)
from ..tensor_ops import split_normalized

if HAVE_SAGE:
    from sage.all import latex as _latex
else:
    from sympy import latex as _latex

__all__ = [
    "HAVE_SAGE",
    "SimplicialException",
    "SymbolicTensor",
    "check_symbolic_corrections",
    "correction_rank",
    "main",
    "n_hypergroupoid_conjecture",
    "test_symbolic_n_hypergroupoid",
]


class SymbolicTensor(_SymbolicTensor):
    """Accept the notebook's tensor_data keyword and retain display helpers.

    Simplicial operations, degeneracy-span membership, correction ranks,
    and horn uniqueness use the shared package implementation.
    """

    def __init__(self, shape, tensor_data=None, init_type="range", *, tensor=None):
        if tensor is not None:
            if tensor_data is not None:
                raise TypeError("Supply either tensor_data or tensor, not both.")
            tensor_data = tensor
        shape = tuple(map(int, shape))
        if tensor_data is not None and np.shape(tensor_data) != shape:
            raise ValueError("The supplied tensor data must match the stated shape.")
        super().__init__(shape, tensor=tensor_data, init_type=init_type)

    @classmethod
    def from_tensor(cls, tensor):
        array = np.asarray(tensor, dtype=object)
        return cls(array.shape, tensor_data=array)

    def __add__(self, other):
        result = super().__add__(other)
        return NotImplemented if result is NotImplemented else self.from_tensor(result.tensor)

    def __sub__(self, other):
        result = super().__sub__(other)
        return NotImplemented if result is NotImplemented else self.from_tensor(result.tensor)

    def order(self):
        """Return the number of axes."""
        return len(self.shape)

    def to_latex(self):
        """Render a two-axis tensor as a LaTeX matrix."""
        if self.order() != 2:
            return "LaTeX representation only available for 2D tensors."
        rows = [" & ".join(_latex(entry) for entry in row) for row in self.tensor]
        return r"\begin{bmatrix}" + "\n" + (r" \\" + "\n").join(rows) + "\n" + r"\end{bmatrix}"

    def decompose_degen(self):
        """Return (u, [(a_i,i)]) with self = u + sum_i s_i(a_i).

        Here u is the normalized projection. All terms a_i lie one degree
        below self. Membership in the degeneracy span means u is zero.
        """
        normalized, terms = split_normalized(self.tensor)
        return self.from_tensor(normalized), [(self.from_tensor(term), i)
                                              for i, term in enumerate(terms)]


def main():
    tensor = SymbolicTensor((3, 3))
    selected = tensor.filler(tensor.horn(1), 1)
    print(f"shape={tensor.shape}, omitted face=1, correction rank={correction_rank(tensor, selected)}")
    print(f"unique horn fillers={tensor.n_hypergroupoid_comparison(outer_horns=True)}")
    edge = SymbolicTensor((2, 2), tensor_data=[[0, 1], [0, 0]])
    degenerate_sum = edge.degen(0) + edge.degen(1)
    print(f"sum of degeneracies belongs to degeneracy span={degenerate_sum.is_degen()}")


if __name__ == "__main__":
    main()
