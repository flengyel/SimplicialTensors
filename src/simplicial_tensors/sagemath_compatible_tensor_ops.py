"""SageMath-compatible symbolic tensor operations.

This module is intended for use under ``sage -python``. When SageMath is not
available, it falls back to SymPy so the module remains importable in standard
Python environments.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Union

import numpy as np

from .tensor_ops import SimplicialException, n_hypergroupoid_conjecture

try:  # pragma: no cover - exercised in Sage environments
    from sage.all import simplify as symbolic_simplify
    from sage.all import var as symbolic_var

    HAVE_SAGE = True
except Exception:  # pragma: no cover - fallback path for non-Sage environments
    import sympy as sp

    HAVE_SAGE = False

    def symbolic_var(name: str):
        return sp.Symbol(name)

    def symbolic_simplify(expr):
        return sp.simplify(expr)


__all__ = [
    "HAVE_SAGE",
    "SymbolicTensor",
    "SimplicialException",
    "n_hypergroupoid_conjecture",
    "correction_rank",
    "test_symbolic_n_hypergroupoid",
    "check_symbolic_corrections",
    "main",
]


def _is_zero(expr: Any) -> bool:
    """Return True when ``expr`` simplifies to zero."""
    simplified = symbolic_simplify(expr)
    try:
        return bool(simplified == 0)
    except Exception:
        is_zero = getattr(simplified, "is_zero", None)
        if callable(is_zero):
            try:
                value = is_zero()
                if value is not None:
                    return bool(value)
            except Exception:
                return False
        return False


class SymbolicTensor:
    def __init__(self, shape: Tuple[int, ...], tensor=None, init_type: str = "range"):
        self.shape = tuple(shape)

        if tensor is not None:
            self.tensor = np.array(tensor, dtype=object, copy=True)
        else:
            self.tensor = np.empty(self.shape, dtype=object)
            for idx in np.ndindex(self.shape):
                idx_str = "_".join(map(str, idx))
                if init_type == "range":
                    self.tensor[idx] = symbolic_var(f"x_{idx_str}")
                elif init_type == "zeros":
                    self.tensor[idx] = 0
                elif init_type == "ones":
                    self.tensor[idx] = 1
                else:
                    raise ValueError(f"Unsupported init_type: {init_type}")

    def __add__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Cannot add tensors of different shapes {self.shape} vs {other.shape}")
        out = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            out[idx] = self.tensor[idx] + other.tensor[idx]
        return SymbolicTensor(self.shape, tensor=out)

    def __sub__(self, other: "SymbolicTensor") -> "SymbolicTensor":
        if not isinstance(other, SymbolicTensor):
            return NotImplemented
        if self.shape != other.shape:
            raise ValueError(f"Cannot subtract tensors of different shapes {self.shape} vs {other.shape}")
        out = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            out[idx] = self.tensor[idx] - other.tensor[idx]
        return SymbolicTensor(self.shape, tensor=out)

    @staticmethod
    def from_tensor(tensor) -> "SymbolicTensor":
        return SymbolicTensor(tuple(tensor.shape), tensor=tensor)

    def dimen(self) -> int:
        return min(self.shape) - 1

    def _dims(self):
        return tuple(np.arange(dim_size) for dim_size in self.shape)

    def face(self, i: int) -> "SymbolicTensor":
        d = min(self.shape)
        if not (0 <= i < d):
            raise IndexError(f"Face index {i} out of bounds for simplicial dimension {d}")

        axes = self._dims()
        indices = [np.delete(axes[dim], i) for dim in range(len(self.shape))]
        grid = np.ix_(*indices)
        result = self.tensor[grid]
        return SymbolicTensor(result.shape, tensor=result)

    def degen(self, k: int) -> "SymbolicTensor":
        result = self.tensor
        for axis in range(result.ndim):
            slices: list[Union[int, slice]] = [slice(None)] * result.ndim
            slices[axis] = k
            insert_slice = result[tuple(slices)]
            result = np.insert(result, k, insert_slice, axis=axis)
        return SymbolicTensor(result.shape, tensor=result)

    def bdry(self) -> "SymbolicTensor":
        d = min(self.shape)
        result_shape = tuple(dim - 1 for dim in self.shape)
        result = np.empty(result_shape, dtype=object)
        for idx in np.ndindex(result_shape):
            result[idx] = 0

        for i in range(d):
            face_i = self.face(i)
            for idx in np.ndindex(result_shape):
                if i % 2 == 0:
                    result[idx] += face_i.tensor[idx]
                else:
                    result[idx] -= face_i.tensor[idx]

        return SymbolicTensor(result_shape, tensor=result)

    def horn(self, k: int) -> List["SymbolicTensor"]:
        d = self.dimen() + 1
        if not (0 <= k < d):
            raise ValueError(f"Horn index {k} must be in [0, {d - 1}]")

        faces = []
        zero_shape = tuple(dim - 1 for dim in self.shape)
        for i in range(d):
            if i == k:
                zero_tensor = np.empty(zero_shape, dtype=object)
                for idx in np.ndindex(zero_shape):
                    zero_tensor[idx] = 0
                faces.append(SymbolicTensor(zero_shape, tensor=zero_tensor))
            else:
                faces.append(self.face(i))
        return faces

    def filler(self, horn_list: List["SymbolicTensor"], k: int) -> "SymbolicTensor":
        g = horn_list[k].degen(0)

        for r in range(k):
            face_gr = g.face(r)
            diff_tensor = np.empty(face_gr.shape, dtype=object)
            for idx in np.ndindex(face_gr.shape):
                diff_tensor[idx] = face_gr.tensor[idx] - horn_list[r].tensor[idx]
            degen_diff = SymbolicTensor(face_gr.shape, tensor=diff_tensor).degen(r)
            for idx in np.ndindex(g.shape):
                g.tensor[idx] = g.tensor[idx] - degen_diff.tensor[idx]

        t = len(horn_list) - 1
        while t > k:
            face_gt = g.face(t)
            diff_tensor = np.empty(face_gt.shape, dtype=object)
            for idx in np.ndindex(face_gt.shape):
                diff_tensor[idx] = horn_list[t].tensor[idx] - face_gt.tensor[idx]
            degen_diff = SymbolicTensor(face_gt.shape, tensor=diff_tensor).degen(t - 1)
            for idx in np.ndindex(g.shape):
                g.tensor[idx] = g.tensor[idx] + degen_diff.tensor[idx]
            t -= 1

        return g

    def is_degen(self) -> bool:
        d = self.dimen()
        for i in range(d):
            face_i = self.face(i)
            degen_i = face_i.degen(i)
            if all(_is_zero(self.tensor[idx] - degen_i.tensor[idx]) for idx in np.ndindex(self.shape)):
                return True
        return False

    def n_hypergroupoid_comparison(self, outer_horns=False, verbose=False, allow_degen=False) -> bool:
        boundary = self.bdry()
        if not allow_degen and boundary.is_degen():
            if verbose:
                print("Boundary is degenerate.")
            raise SimplicialException("Degenerate boundary.")

        dim = self.dimen()
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
                    diff = symbolic_simplify(original.tensor[idx] - reproduced.tensor[idx])
                    if not _is_zero(diff):
                        if verbose:
                            print(f"Disagreement at face {j}, index {idx}: {diff}")
                        raise SimplicialException(
                            f"Original horn and filler horn disagree at face {j}, position {idx}."
                        )

            differences = []
            for idx in np.ndindex(self.shape):
                diff = symbolic_simplify(self.tensor[idx] - filler_i.tensor[idx])
                if not _is_zero(diff):
                    differences.append((idx, self.tensor[idx], filler_i.tensor[idx]))

            if differences:
                if verbose:
                    print("Multiple fillers exist. The original tensor and the filler differ at the following indices:")
                    for idx, orig, fill in differences:
                        print(f"  At index {idx}:")
                        print(f"    Original: {orig}")
                        print(f"    Filler:   {fill}")
                return False

        if verbose:
            print("Unique filler.")
        return True

    def simplify(self) -> "SymbolicTensor":
        for idx in np.ndindex(self.shape):
            self.tensor[idx] = symbolic_simplify(self.tensor[idx])
        return self

    def subs(self, substitutions: dict) -> "SymbolicTensor":
        for idx in np.ndindex(self.shape):
            value = self.tensor[idx]
            subs_method = getattr(value, "subs", None)
            if callable(subs_method):
                self.tensor[idx] = subs_method(substitutions)
            else:
                self.tensor[idx] = value
        return self

    def __str__(self) -> str:
        return str(self.tensor)

    def __repr__(self) -> str:
        return f"SymbolicTensor(shape={self.shape})"


def correction_rank(original: SymbolicTensor, filler: SymbolicTensor) -> int:
    if original.shape != filler.shape:
        raise ValueError("Tensors must have the same shape to compare.")

    differences = set()
    for idx in np.ndindex(original.shape):
        diff = symbolic_simplify(original.tensor[idx] - filler.tensor[idx])
        if not _is_zero(diff):
            differences.add(str(diff))
    return len(differences)


def test_symbolic_n_hypergroupoid(shape: Tuple[int, ...], verbose=True):
    sym_tensor = SymbolicTensor(shape)
    conjecture = n_hypergroupoid_conjecture(shape, verbose=verbose)

    try:
        comparison = sym_tensor.n_hypergroupoid_comparison(outer_horns=True, verbose=verbose)
        if verbose:
            print(f"Conjecture predicts unique fillers: {conjecture}")
            print(f"Filler uniqueness observed: {comparison}")
            if conjecture == comparison:
                print("PASS: The n-hypergroupoid conjecture is confirmed for this shape.")
            else:
                print("FAIL: Observation does not match conjecture prediction.")
        return conjecture, comparison, sym_tensor
    except SimplicialException as exc:
        if "Degenerate boundary" in str(exc):
            if verbose:
                print("Skipping comparison due to degenerate boundary.")
            return conjecture, None, sym_tensor
        raise


def check_symbolic_corrections(t, t_prime, horn_faces, k):
    shape = t.shape
    n = t.dimen()

    print(f"Checking horn({n},{k}) indices missing from symbolic tensor with shape {shape}.")

    all_symbols = set()
    for idx in np.ndindex(shape):
        expr = t.tensor[idx]
        if not _is_zero(expr):
            all_symbols.add(str(expr))

    face_symbol_union = set()
    for face_idx, face in enumerate(horn_faces):
        if face_idx == k:
            continue
        for subidx in np.ndindex(face.shape):
            expr = face.tensor[subidx]
            if not _is_zero(expr):
                face_symbol_union.add(str(expr))

    missing_symbols = all_symbols - face_symbol_union

    changed_symbols = set()
    for idx in np.ndindex(shape):
        expr_orig = t.tensor[idx]
        expr_new = t_prime.tensor[idx]
        diff_expr = symbolic_simplify(expr_new - expr_orig)
        if not _is_zero(diff_expr):
            if _is_zero(expr_orig):
                changed_symbols.add(str(expr_new))
            else:
                changed_symbols.add(str(expr_orig))

    if changed_symbols == missing_symbols:
        print(f"Success: the filler differed from the original at {len(missing_symbols)} indices.")
        return True

    print("Mismatch in correction terms vs. missing symbols.")
    extra = changed_symbols - missing_symbols
    missed = missing_symbols - changed_symbols
    if extra:
        print("Symbols changed that were not missing:", extra)
    if missed:
        print("Symbols missing but unchanged:", missed)
    return False


def main():
    shape = (3, 3)
    conjecture, comparison, sym_tensor = test_symbolic_n_hypergroupoid(shape)

    horn_1 = sym_tensor.horn(1)
    filler_1 = sym_tensor.filler(horn_1, 1)

    print("Original tensor:")
    print(sym_tensor)
    print("\nFiller tensor:")
    print(filler_1)

    print("\nComparison of original and filler tensors:")
    result = check_symbolic_corrections(sym_tensor, filler_1, horn_1, 1)
    print("Check result:", result)
    print(f"Shape: {shape}, Conjecture: {conjecture}, Comparison: {comparison}")


if __name__ == "__main__":
    main()