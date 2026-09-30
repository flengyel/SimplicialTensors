"""Sparse rational tensors in a fixed rank-shifted simplicial module.

For fixed nonnegative shifts ``c = (c_1, ..., c_k)``, degree ``p`` has
axis lengths ``p + c_a + 1``.  Only indices ``0, ..., p`` index faces
and degeneracies: additional tail coordinates do not add simplicial maps.

Coefficients are exact rational numbers.  Storage and all operations use
only nonzero entries; no dense ambient tensor is allocated.  A degeneracy
can nevertheless expand one entry into as many as ``2**k`` entries.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from types import MappingProxyType

Index = tuple[int, ...]
Coefficient = int | Fraction

__all__ = [
    "SparseTensor",
    "GraphEmbedding",
    "graph_embedding",
    "petersen_edges",
    "circle_generator",
]


def _nonnegative_integer(value: int, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _rational(value: Coefficient) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError("coefficients must be int or Fraction; floats are not exact input")
    return Fraction(value)


def _shifts(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("a tensor must have at least one axis")
    return tuple(_nonnegative_integer(value, "axis shift") for value in result)


@dataclass(frozen=True)
class SparseTensor:
    """An immutable sparse tensor over ``Fraction`` in a specified degree.

    ``entries`` is copied, zero coefficients are removed, and the resulting
    mapping is read-only.  Coordinates are zero-based.  The coefficient of
    an omitted coordinate is zero.
    """

    degree: int
    shifts: tuple[int, ...]
    entries: Mapping[Index, Fraction]

    def __post_init__(self) -> None:
        _nonnegative_integer(self.degree, "degree")
        shifts = _shifts(self.shifts)
        object.__setattr__(self, "shifts", shifts)
        if not isinstance(self.entries, Mapping):
            raise TypeError("entries must be a mapping from index tuples to coefficients")
        entries: dict[Index, Fraction] = {}
        for index, coefficient in self.entries.items():
            index = self._checked_index(index)
            value = _rational(coefficient)
            if value:
                entries[index] = value
        object.__setattr__(self, "entries", MappingProxyType(entries))

    @classmethod
    def zero(cls, degree: int, shifts: Iterable[int]) -> SparseTensor:
        """Return the zero tensor with the specified type."""
        return cls(degree, tuple(shifts), {})

    @classmethod
    def unit(
        cls,
        degree: int,
        shifts: Iterable[int],
        index: Index,
        coefficient: Coefficient = 1,
    ) -> SparseTensor:
        """Return a scalar multiple of a coordinate tensor unit."""
        return cls(degree, tuple(shifts), {index: _rational(coefficient)})

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.degree + shift + 1 for shift in self.shifts)

    @property
    def nnz(self) -> int:
        """Number of nonzero stored coefficients."""
        return len(self.entries)

    def _checked_index(self, index: Index) -> Index:
        if not isinstance(index, tuple) or len(index) != len(self.shifts):
            raise ValueError("an index must be a tuple with one coordinate per axis")
        for coordinate, length in zip(index, self.shape):
            _nonnegative_integer(coordinate, "coordinate")
            if coordinate >= length:
                raise ValueError(f"coordinate {coordinate} is outside axis length {length}")
        return index

    def coefficient(self, index: Index) -> Fraction:
        """Return a coefficient, rejecting coordinates outside this tensor."""
        return self.entries.get(self._checked_index(index), Fraction(0))

    def __bool__(self) -> bool:
        return bool(self.entries)

    def __hash__(self) -> int:
        return hash((self.degree, self.shifts, tuple(sorted(self.entries.items()))))

    def _check_same_type(self, other: SparseTensor) -> None:
        if self.degree != other.degree or self.shifts != other.shifts:
            raise ValueError("tensor degrees and axis shifts must agree")

    def __add__(self, other: SparseTensor) -> SparseTensor:
        if not isinstance(other, SparseTensor):
            return NotImplemented
        self._check_same_type(other)
        entries = dict(self.entries)
        for index, value in other.entries.items():
            entries[index] = entries.get(index, Fraction(0)) + value
        return SparseTensor(self.degree, self.shifts, entries)

    def __radd__(self, other: object) -> SparseTensor:
        # Support the standard ``sum(tensors)`` convention.
        if isinstance(other, int) and not isinstance(other, bool) and other == 0:
            return self
        return NotImplemented

    def __neg__(self) -> SparseTensor:
        return self * -1

    def __sub__(self, other: SparseTensor) -> SparseTensor:
        if not isinstance(other, SparseTensor):
            return NotImplemented
        return self + (-other)

    def __mul__(self, scalar: Coefficient) -> SparseTensor:
        scalar = _rational(scalar)
        return SparseTensor(
            self.degree,
            self.shifts,
            {index: scalar * value for index, value in self.entries.items()},
        )

    def __rmul__(self, scalar: Coefficient) -> SparseTensor:
        return self * scalar

    def __truediv__(self, scalar: Coefficient) -> SparseTensor:
        scalar = _rational(scalar)
        if not scalar:
            raise ZeroDivisionError("division by zero")
        return self * (1 / scalar)

    def _check_simplicial_index(self, index: int) -> None:
        _nonnegative_integer(index, "simplicial index")
        if index > self.degree:
            raise ValueError(f"simplicial index must lie in 0, ..., {self.degree}")

    def face(self, index: int) -> SparseTensor:
        """Delete ``index`` on every axis, giving degree ``p - 1``.

        Faces in degree zero are intentionally unavailable: this class
        models the unaugmented simplicial module.
        """
        if self.degree == 0:
            raise ValueError("degree-zero tensors have no unaugmented face")
        self._check_simplicial_index(index)
        entries = {
            tuple(coordinate - (coordinate > index) for coordinate in position): value
            for position, value in self.entries.items()
            if index not in position
        }
        return SparseTensor(self.degree - 1, self.shifts, entries)

    def degeneracy(self, index: int) -> SparseTensor:
        """Duplicate ``index`` on every axis, giving degree ``p + 1``."""
        self._check_simplicial_index(index)
        entries: dict[Index, Fraction] = {}
        for position, value in self.entries.items():
            choices = (
                (index, index + 1)
                if coordinate == index
                else (coordinate + (coordinate > index),)
                for coordinate in position
            )
            for expanded in product(*choices):
                entries[expanded] = value
        return SparseTensor(self.degree + 1, self.shifts, entries)

    def boundary(self) -> SparseTensor:
        """Return ``sum((-1)**i * d_i(self), i=0,...,p)`` for ``p > 0``.

        The unaugmented chain differential out of degree zero is the map
        to the zero module, which has no ``SparseTensor`` representation;
        this method raises ``ValueError`` there.
        """
        if self.degree == 0:
            raise ValueError("the degree-zero boundary targets the omitted zero module")
        result = SparseTensor.zero(self.degree - 1, self.shifts)
        for index in range(self.degree + 1):
            result = result + (-1) ** index * self.face(index)
        return result

    def normalize(self) -> SparseTensor:
        """Project onto ``N_p = intersection(ker(d_i), i=1,...,p)``.

        Apply ``1 - s_(i-1) d_i`` in descending order ``i=p,...,1``.
        On this normalized submodule the chain differential is ``d_0``.
        """
        result = self
        for index in range(self.degree, 0, -1):
            result = result - result.face(index).degeneracy(index - 1)
        return result

    def outer(self, other: SparseTensor) -> SparseTensor:
        """Concatenate axes in a degreewise tensor product.

        The factors must have the same simplicial degree; their shifts
        need not agree.  Faces and degeneracies commute with this product.
        """
        if not isinstance(other, SparseTensor):
            raise TypeError("the other factor must be a SparseTensor")
        if self.degree != other.degree:
            raise ValueError("outer factors must have the same simplicial degree")
        entries = {
            left + right: left_value * right_value
            for left, left_value in self.entries.items()
            for right, right_value in other.entries.items()
        }
        return SparseTensor(self.degree, self.shifts + other.shifts, entries)

    def pad_shifts(self, new_shifts: Iterable[int]) -> SparseTensor:
        """Embed by zero extension into larger trailing coordinate sets."""
        new_shifts = _shifts(new_shifts)
        if len(new_shifts) != len(self.shifts):
            raise ValueError("padding must preserve the number of axes")
        if any(new < old for new, old in zip(new_shifts, self.shifts)):
            raise ValueError("padding cannot decrease an axis shift")
        return SparseTensor(self.degree, new_shifts, self.entries)


@dataclass(frozen=True)
class GraphEmbedding:
    """Generators embedding an oriented finite graph's rational chains.

    Vertices are labeled ``1,...,vertex_count``.  The ordered edges may
    include loops and repeated pairs.  Tuple entries ``vertices[v-1]``
    and ``edge_tensors[j-1]`` encode vertex ``v`` and edge ``j``.
    """

    vertex_count: int
    edges: tuple[tuple[int, int], ...]
    shift: int
    vertices: tuple[SparseTensor, ...]
    edge_tensors: tuple[SparseTensor, ...]

    @property
    def normalized_edges(self) -> tuple[SparseTensor, ...]:
        return tuple(tensor.normalize() for tensor in self.edge_tensors)


def graph_embedding(
    vertex_count: int,
    edges: Iterable[tuple[int, int]],
    shift: int | None = None,
) -> GraphEmbedding:
    """Encode any finite oriented graph in a three-axis shifted DSTM.

    For edge ``j: u -> v`` and ``r >= max(vertex_count, edge_count)``, use

    ``a_v = E_(0,v,v)`` and
    ``T_j = E_(0,u+1,u+1) + E_(1,v+1,v+1) + E_(0,1,j+1)``.

    Thus ``d_1(T_j) = a_u`` and ``d_0(T_j) = a_v``.  The independent
    marker coordinate ``(0,1,j+1)`` ensures that normalized edge classes
    have no extra linear relations.  The simplicial submodule generated
    by these vertices and edges has the graph's normalized chain complex.
    """
    vertex_count = _nonnegative_integer(vertex_count, "vertex_count")
    edge_list = []
    for edge in edges:
        edge = tuple(edge)
        if len(edge) != 2:
            raise ValueError("each edge must have a source and a target")
        for vertex in edge:
            _nonnegative_integer(vertex, "vertex label")
            if not 1 <= vertex <= vertex_count:
                raise ValueError("vertex labels must lie in 1, ..., vertex_count")
        edge_list.append(edge)
    edge_list = tuple(edge_list)
    minimum = max(vertex_count, len(edge_list))
    if shift is None:
        shift = minimum
    shift = _nonnegative_integer(shift, "shift")
    if shift < minimum:
        raise ValueError("shift must be at least max(vertex_count, edge_count)")
    shifts = (shift, shift, shift)
    vertices = tuple(
        SparseTensor.unit(0, shifts, (0, vertex, vertex))
        for vertex in range(1, vertex_count + 1)
    )
    tensors = tuple(
        SparseTensor.unit(1, shifts, (0, source + 1, source + 1))
        + SparseTensor.unit(1, shifts, (1, target + 1, target + 1))
        + SparseTensor.unit(1, shifts, (0, 1, edge_number + 1))
        for edge_number, (source, target) in enumerate(edge_list, 1)
    )
    return GraphEmbedding(vertex_count, edge_list, shift, vertices, tensors)


def petersen_edges() -> tuple[tuple[int, int], ...]:
    """Return Petersen edges with vertices the lexicographic 2-subsets of 5.

    Labels ``1,...,10`` index ``combinations(range(5), 2)``; disjoint
    pairs are adjacent.  Each edge is oriented from its smaller label.
    """
    vertices = tuple(frozenset(pair) for pair in combinations(range(5), 2))
    return tuple(
        (left + 1, right + 1)
        for left, right in combinations(range(len(vertices)), 2)
        if vertices[left].isdisjoint(vertices[right])
    )


def circle_generator() -> SparseTensor:
    """Return ``E_(0,1)``, generating the reduced rational simplicial circle.

    Both faces are zero.  Its generated simplicial submodule has one
    normalized generator in degree one and none in other degrees.
    """
    return SparseTensor.unit(1, (0, 0), (0, 1))
