"""Finite chain complexes and basis calculations over Q or a prime field.

Matrices act on column vectors. Differentials are indexed by their source
degree, including the zero matrix from degree zero to the zero space.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from fractions import Fraction
from numbers import Rational
from typing import cast

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix

__all__ = ["ChainComplex", "ExactField"]


def _nonnegative_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return value


@dataclass(frozen=True)
class ExactField:
    """Exact arithmetic over Q, or F_p for a supplied prime modulus.

    Matrix entries are Python/NumPy integers, Fraction, SymPy rationals,
    or exact Rational objects providing as_integer_ratio (including Sage QQ).
    Finite-field matrices store canonical integer residues. Use the field's
    operations when multiplying or taking ranks of those stored matrices.
    """

    modulus: int | None = None

    def __post_init__(self):
        if self.modulus is not None:
            try:
                modulus = _nonnegative_integer(self.modulus, "The modulus")
            except (TypeError, ValueError) as exc:
                raise ValueError("The modulus must be prime.") from exc
            if not sp.isprime(modulus):
                raise ValueError("The modulus must be prime.")
            object.__setattr__(self, "modulus", modulus)

    @property
    def domain(self):
        return sp.QQ if self.modulus is None else sp.GF(self.modulus)

    def scalar(self, value):
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        if isinstance(value, (int, np.integer, Fraction, sp.Rational)):
            rational = sp.Rational(value)
        elif isinstance(value, Rational):
            try:
                numerator, denominator = value.as_integer_ratio()
                rational = sp.Rational(operator.index(numerator), operator.index(denominator))
            except (AttributeError, TypeError) as exc:
                raise TypeError("Exact coefficients must be integers or rationals.") from exc
        else:
            raise TypeError("Exact coefficients must be integers or rationals.")
        if self.modulus is None:
            return rational
        try:
            inverse = pow(int(rational.q), -1, self.modulus)
        except ValueError as exc:
            raise ValueError("A coefficient denominator vanishes in the field.") from exc
        return sp.Integer(int(rational.p) * inverse % self.modulus)

    def matrix(self, values) -> sp.ImmutableMatrix:
        """Copy exact entries into an immutable matrix and reduce in the field.

        A one-dimensional sequence is a column vector; an empty sequence is
        a 0-by-0 matrix. Use an explicitly shaped matrix for other empty shapes.
        """
        if isinstance(values, sp.MatrixBase):
            rows, cols = values.shape
            entries = list(values)
        else:
            array = np.asarray(values, dtype=object)
            if array.ndim == 1:
                rows, cols = (array.size, 1) if array.size else (0, 0)
            elif array.ndim == 2:
                rows, cols = array.shape
            else:
                raise ValueError("A matrix needs one or two dimensions.")
            entries = list(array.flat)
        return sp.ImmutableMatrix(rows, cols, [self.scalar(value) for value in entries])

    def _domain_matrix(self, values):
        if isinstance(values, (sp.MutableSparseMatrix, sp.ImmutableSparseMatrix)):
            domain = self.domain
            rows = {}
            for (row, column), value in values.todok().items():
                value = self.scalar(value)
                if value:
                    rows.setdefault(row, {})[column] = domain.convert(value)
            return DomainMatrix(rows, values.shape, domain)
        return DomainMatrix.from_Matrix(self.matrix(values)).convert_to(self.domain)

    def rank(self, values) -> int:
        return int(self._domain_matrix(values).rank())

    def rref(self, values):
        reduced, pivots = self._domain_matrix(values).rref()
        return self.matrix(reduced.to_Matrix()), tuple(int(pivot) for pivot in pivots)

    def nullspace(self, values) -> sp.ImmutableMatrix:
        """Return a matrix whose columns form the kernel basis."""
        basis = self._domain_matrix(values).nullspace().transpose().to_Matrix()
        return self.matrix(basis)

    def column_basis(self, values) -> sp.ImmutableMatrix:
        values = self.matrix(values)
        pivots = self.rref(values)[1]
        return self.matrix(values.extract(range(values.rows), pivots))

    def matmul(self, left, right) -> sp.ImmutableMatrix:
        left, right = self._domain_matrix(left), self._domain_matrix(right)
        return self.matrix((left * right).to_Matrix())

    def coordinates(self, basis, targets) -> sp.ImmutableMatrix:
        """Solve basis * X = targets for a basis with independent columns.

        Reject dependent columns or targets outside the column span.
        """
        basis, targets = self.matrix(basis), self.matrix(targets)
        if basis.rows != targets.rows:
            raise ValueError("Basis and targets must have the same number of rows.")
        if self.rank(basis) != basis.cols:
            raise ValueError("Basis columns must be independent in the coefficient field.")
        reduced, pivots = self.rref(basis.row_join(targets))
        if pivots != tuple(range(basis.cols)):
            raise ValueError("Targets lie outside the basis column span.")
        return self.matrix(reduced[:basis.cols, basis.cols:])


@dataclass(frozen=True)
class ChainComplex:
    """A finite based chain complex with optional representatives in ambient bases.

    ``differentials[n]`` has shape ``(dimensions[n-1], dimensions[n])``
    for n > 0, and differential zero has shape ``(0, dimensions[0])``.
    ``lifts[n]`` maps chain coordinates to the rows labeled by
    ``ambient_bases[n]``. Each lift has independent columns; identity lifts
    and standard coordinate labels are supplied when omitted. Ambient lifts
    need not intertwine a separate ambient differential (for example, lifts
    of quotient classes are representatives modulo the quotient subcomplex).
    """

    dimensions: tuple[int, ...]
    differentials: tuple[sp.ImmutableMatrix, ...]
    modulus: int | None = None
    lifts: tuple[sp.ImmutableMatrix, ...] | None = None
    ambient_bases: tuple[tuple[object, ...], ...] | None = None

    def __post_init__(self):
        coefficient_field = ExactField(self.modulus)
        dimensions = tuple(_nonnegative_integer(value, "Chain dimension") for value in self.dimensions)
        differentials = tuple(coefficient_field.matrix(value) for value in self.differentials)
        if len(differentials) != len(dimensions):
            raise ValueError("Supply one differential for each chain degree, including zero.")
        for degree, (dimension, differential) in enumerate(zip(dimensions, differentials)):
            expected = (dimensions[degree - 1] if degree else 0, dimension)
            if differential.shape != expected:
                raise ValueError(f"Differential {degree} has shape {differential.shape}, expected {expected}.")
            if degree and any(coefficient_field.matmul(differentials[degree - 1], differential)):
                raise ValueError("Consecutive differentials must compose to zero in the coefficient field.")
        lifts = (tuple(sp.ImmutableMatrix.eye(dimension) for dimension in dimensions)
                 if self.lifts is None else tuple(coefficient_field.matrix(value) for value in self.lifts))
        if len(lifts) != len(dimensions):
            raise ValueError("Supply one lift matrix for each chain degree.")
        for dimension, lift in zip(dimensions, lifts):
            if lift.cols != dimension or coefficient_field.rank(lift) != dimension:
                raise ValueError("Each lift must have one independent column per chain basis element.")
        labels = (tuple(tuple(range(lift.rows)) for lift in lifts) if self.ambient_bases is None
                  else tuple(tuple(basis) for basis in self.ambient_bases))
        if len(labels) != len(dimensions) or any(len(basis) != lift.rows for basis, lift in zip(labels, lifts)):
            raise ValueError("Ambient basis labels must match the rows of each lift matrix.")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(self, "differentials", differentials)
        object.__setattr__(self, "modulus", coefficient_field.modulus)
        object.__setattr__(self, "lifts", lifts)
        object.__setattr__(self, "ambient_bases", labels)

    @property
    def field(self) -> ExactField:
        return ExactField(self.modulus)

    @property
    def chain_dimensions(self):
        return self.dimensions

    def dimension(self, degree: int) -> int:
        degree = _nonnegative_integer(degree, "Degree")
        return self.dimensions[degree] if degree < len(self.dimensions) else 0

    def differential(self, degree: int) -> sp.ImmutableMatrix:
        degree = _nonnegative_integer(degree, "Degree")
        if degree < len(self.differentials):
            return self.differentials[degree]
        return sp.ImmutableMatrix.zeros(self.dimension(degree - 1) if degree else 0, 0)

    def homology(self):
        """Return the existing dimension/rank summary in ascending degrees."""
        from .homology import GeneratedHomology

        ranks = tuple(self.field.rank(differential) for differential in self.differentials)
        incoming = ranks[1:] + (0,) if ranks else ()
        betti = tuple(dimension - outgoing - following
                      for dimension, outgoing, following in zip(self.dimensions, ranks, incoming))
        return GeneratedHomology(self.dimensions, ranks, betti)

    def cycles(self, degree: int) -> sp.ImmutableMatrix:
        return self.field.nullspace(self.differential(degree))

    def boundaries(self, degree: int) -> sp.ImmutableMatrix:
        _nonnegative_integer(degree, "Degree")
        return self.field.column_basis(self.differential(degree + 1))

    def homology_representatives(self, degree: int) -> sp.ImmutableMatrix:
        """Return cycle columns forming a basis modulo the boundary subspace."""
        cycles, boundaries = self.cycles(degree), self.boundaries(degree)
        pivots = self.field.rref(boundaries.row_join(cycles))[1]
        selected = [pivot - boundaries.cols for pivot in pivots if pivot >= boundaries.cols]
        return self.field.matrix(cycles.extract(range(cycles.rows), selected))

    def lift(self, degree: int, coordinates) -> sp.ImmutableMatrix:
        degree = _nonnegative_integer(degree, "Degree")
        coordinates = self.field.matrix(coordinates)
        if coordinates.rows != self.dimension(degree):
            raise ValueError("Coordinates must have one row per chain basis element.")
        if degree >= len(self.dimensions):
            return sp.ImmutableMatrix.zeros(0, coordinates.cols)
        lifts = cast(tuple[sp.ImmutableMatrix, ...], self.lifts)
        return self.field.matmul(lifts[degree], coordinates)
