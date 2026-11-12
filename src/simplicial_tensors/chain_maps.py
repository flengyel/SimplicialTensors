"""Construct exact finite chain maps, embeddings, and quotient-isomorphism lifts.

Matrices act on chain-coordinate columns over Q or a declared prime field.
Ambient representatives stored on ChainComplex objects do not enter these maps.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .chain_complex import ChainComplex, ExactField

__all__ = [
    "CapacityError", "ChainMap", "identity_chain_map", "chain_embedding",
    "chain_isomorphism", "lift_quotient_isomorphism",
]


def _same_complex(left: ChainComplex, right: ChainComplex) -> bool:
    """Compare the based chain data, allowing zero trailing terms."""
    if left.modulus != right.modulus:
        return False
    length = max(len(left.dimensions), len(right.dimensions))
    return all(left.dimension(q) == right.dimension(q)
               and left.differential(q) == right.differential(q) for q in range(length))


def _same_map(left: ChainMap, right: ChainMap) -> bool:
    return (_same_complex(left.source, right.source)
            and _same_complex(left.target, right.target)
            and all(left.matrix(q) == right.matrix(q)
                    for q in range(max(len(left.matrices), len(right.matrices)))))


@dataclass(frozen=True)
class ChainMap:
    """An immutable degree-zero chain map with explicitly checked matrices.

    ``matrices[q]`` has shape ``(target.dimension(q), source.dimension(q))``.
    Omitted trailing matrices are zero. Composition ``f.compose(g)`` means
    f after g. Composability compares chain dimensions and differentials;
    optional ambient labels and lifts do not change chain coordinates.
    """

    source: ChainComplex
    target: ChainComplex
    matrices: tuple[sp.ImmutableMatrix, ...]

    def __post_init__(self):
        if not isinstance(self.source, ChainComplex) or not isinstance(self.target, ChainComplex):
            raise TypeError("Source and target must be ChainComplex objects.")
        if self.source.modulus != self.target.modulus:
            raise ValueError("Chain maps require the same coefficient field.")
        field = self.source.field
        length = max(len(self.source.dimensions), len(self.target.dimensions))
        matrices = tuple(field.matrix(value) for value in self.matrices)
        if len(matrices) > length:
            raise ValueError("Too many chain-map matrices for these complexes.")
        matrices += tuple(sp.ImmutableMatrix.zeros(self.target.dimension(q), self.source.dimension(q))
                          for q in range(len(matrices), length))
        for q, value in enumerate(matrices):
            expected = (self.target.dimension(q), self.source.dimension(q))
            if value.shape != expected:
                raise ValueError(f"Chain-map matrix {q} has shape {value.shape}, expected {expected}.")
            if q:
                left = field.matmul(self.target.differential(q), value)
                right = field.matmul(matrices[q - 1], self.source.differential(q))
                if left != right:
                    raise ValueError(f"Chain-map identity fails in degree {q}.")
        object.__setattr__(self, "matrices", matrices)

    def matrix(self, degree: int) -> sp.ImmutableMatrix:
        """Return one component, extended by zero outside the stored degrees."""
        source_dimension = self.source.dimension(degree)
        target_dimension = self.target.dimension(degree)
        degree = int(degree)
        if degree < len(self.matrices):
            return self.matrices[degree]
        return sp.ImmutableMatrix.zeros(target_dimension, source_dimension)

    def compose(self, other: ChainMap) -> ChainMap:
        """Return self after other."""
        if not isinstance(other, ChainMap):
            raise TypeError("Compose with a ChainMap.")
        if not _same_complex(other.target, self.source):
            raise ValueError("The middle chain complexes do not agree.")
        length = max(len(other.source.dimensions), len(self.target.dimensions))
        return ChainMap(other.source, self.target, tuple(
            self.source.field.matmul(self.matrix(q), other.matrix(q)) for q in range(length)
        ))

    def is_injective(self) -> bool:
        return all(self.source.field.rank(value) == self.source.dimension(q)
                   for q, value in enumerate(self.matrices))

    def is_isomorphism(self) -> bool:
        return all(self.source.dimension(q) == self.target.dimension(q)
                   and self.source.field.rank(value) == self.source.dimension(q)
                   for q, value in enumerate(self.matrices))

    def inverse(self) -> ChainMap:
        """Invert a degreewise isomorphism in the declared field."""
        if not self.is_isomorphism():
            raise ValueError("Only a chain isomorphism has an inverse.")
        return ChainMap(self.target, self.source, tuple(
            self.source.field.coordinates(value, sp.eye(value.rows)) for value in self.matrices
        ))


def identity_chain_map(model: ChainComplex) -> ChainMap:
    return ChainMap(model, model, tuple(sp.eye(size) for size in model.dimensions))


class CapacityError(ValueError):
    """A required boundary or cycle subspace is too large for its target."""

    def __init__(self, degree: int, kind: str, required: int, available: int):
        self.degree = degree
        self.kind = kind
        self.required = required
        self.available = available
        super().__init__(f"Insufficient {kind} capacity in degree {degree}: "
                         f"required {required}, available {available}.")


def _solve(field: ExactField, coefficients, targets) -> sp.ImmutableMatrix:
    """Choose a solution of A*X=B by setting free coordinates to zero."""
    coefficients, targets = field.matrix(coefficients), field.matrix(targets)
    if coefficients.rows != targets.rows:
        raise ValueError("Linear equations and targets have incompatible row counts.")
    reduced, pivots = field.rref(coefficients.row_join(targets))
    if any(pivot >= coefficients.cols for pivot in pivots):
        raise ValueError("The required linear lift does not exist.")
    result = sp.zeros(coefficients.cols, targets.cols)
    for row, pivot in enumerate(pivots):
        for column in range(targets.cols):
            result[pivot, column] = reduced[row, coefficients.cols + column]
    return field.matrix(result)


@dataclass(frozen=True)
class _Basis:
    boundaries: sp.ImmutableMatrix
    homology: sp.ImmutableMatrix
    upper: sp.ImmutableMatrix
    all: sp.ImmutableMatrix


def _adapted_bases(model: ChainComplex, length: int) -> tuple[_Basis, ...]:
    """Use boundary, homology, and paired upper-disk basis columns."""
    field = model.field
    result = []
    for q in range(length):
        boundaries = model.boundaries(q)
        homology = model.homology_representatives(q)
        upper = (_solve(field, model.differential(q), model.boundaries(q - 1))
                 if q else sp.ImmutableMatrix.zeros(model.dimension(q), 0))
        full = field.matrix(boundaries.row_join(homology).row_join(upper))
        if full.cols != model.dimension(q) or field.rank(full) != model.dimension(q):
            raise ArithmeticError("The adapted chain basis is not invertible.")
        result.append(_Basis(boundaries, homology, upper, full))
    return tuple(result)


def chain_embedding(source: ChainComplex, target: ChainComplex) -> ChainMap:
    """Construct a chain injection exactly when boundary and cycle capacities fit.

    In each degree the required inequalities are dim B(source) <= dim B(target)
    and dim Z(source) <= dim Z(target). Homology classes may map to target
    boundaries; this is a degreewise chain injection, not a homology injection.
    A CapacityError identifies the first failing degree and subspace.
    """
    if source.modulus != target.modulus:
        raise ValueError("Chain embeddings require the same coefficient field.")
    length = max(len(source.dimensions), len(target.dimensions))
    source_bases = _adapted_bases(source, length)
    target_bases = _adapted_bases(target, length)
    field = source.field
    matrices = []
    for q, (left, right) in enumerate(zip(source_bases, target_bases)):
        boundaries = left.boundaries.cols
        homology = left.homology.cols
        if boundaries > right.boundaries.cols:
            raise CapacityError(q, "boundary", boundaries, right.boundaries.cols)
        required_cycles = boundaries + homology
        available_cycles = right.boundaries.cols + right.homology.cols
        if required_cycles > available_cycles:
            raise CapacityError(q, "cycle", required_cycles, available_cycles)
        cycle_remainder = right.boundaries[:, boundaries:].row_join(right.homology)
        images = right.boundaries[:, :boundaries].row_join(cycle_remainder[:, :homology])
        images = images.row_join(right.upper[:, :left.upper.cols])
        inverse = field.coordinates(left.all, sp.eye(left.all.rows))
        matrices.append(field.matmul(images, inverse))
    result = ChainMap(source, target, tuple(matrices))
    if not result.is_injective():
        raise ArithmeticError("The constructed chain embedding is not injective.")
    return result


def chain_isomorphism(source: ChainComplex, target: ChainComplex) -> ChainMap:
    """Choose an isomorphism when dimensions and differential ranks agree."""
    if source.modulus != target.modulus:
        raise ValueError("Chain isomorphisms require the same coefficient field.")
    length = max(len(source.dimensions), len(target.dimensions))
    if any(source.dimension(q) != target.dimension(q)
           or source.field.rank(source.differential(q)) != target.field.rank(target.differential(q))
           for q in range(length)):
        raise ValueError("Chain dimensions and differential ranks must agree.")
    return chain_embedding(source, target)


def _require_epimorphism(projection: ChainMap):
    if any(projection.source.field.rank(value) != projection.target.dimension(q)
           for q, value in enumerate(projection.matrices)):
        raise ValueError("Each quotient projection must be degreewise surjective.")


def _require_projective(source: ChainComplex):
    if any(source.homology().homology_dimensions[1:]):
        raise ValueError("Source must be projective as a nonnegative chain complex: "
                         "positive-degree homology must vanish.")


def _projective_lift(projection: ChainMap, morphism: ChainMap) -> ChainMap:
    """Lift along an epimorphism using the source's S0 and disk summands."""
    source, target, field = morphism.source, projection.source, morphism.source.field
    length = max(len(source.dimensions), len(target.dimensions))
    bases = _adapted_bases(source, length + 1)
    tops = [sp.ImmutableMatrix.zeros(target.dimension(0), 0)]
    for q in range(1, length + 1):
        tops.append(_solve(field, projection.matrix(q), field.matmul(morphism.matrix(q), bases[q].upper)))
    matrices = []
    for q in range(length):
        bottom = field.matmul(target.differential(q + 1), tops[q + 1])
        homology = (_solve(field, projection.matrix(0), field.matmul(morphism.matrix(0), bases[0].homology))
                    if q == 0 else sp.ImmutableMatrix.zeros(target.dimension(q), 0))
        images = bottom.row_join(homology).row_join(tops[q])
        inverse = field.coordinates(bases[q].all, sp.eye(source.dimension(q)))
        matrices.append(field.matmul(images, inverse))
    result = ChainMap(source, target, tuple(matrices))
    if not _same_map(projection.compose(result), morphism):
        raise ArithmeticError("The constructed projective lift does not commute with its projection.")
    return result


def _power(field: ExactField, values, exponent: int) -> sp.ImmutableMatrix:
    result = field.matrix(sp.eye(values.rows))
    values = field.matrix(values)
    while exponent:
        if exponent & 1:
            result = field.matmul(result, values)
        exponent //= 2
        if exponent:
            values = field.matmul(values, values)
    return result


def _subcomplex(source: ChainComplex, bases) -> ChainComplex:
    field = source.field
    differentials = []
    for q, basis in enumerate(bases):
        differentials.append(field.coordinates(bases[q - 1], field.matmul(source.differential(q), basis))
                             if q else sp.ImmutableMatrix.zeros(0, basis.cols))
    return ChainComplex(tuple(basis.cols for basis in bases), tuple(differentials),
                        modulus=source.modulus, lifts=tuple(bases))


def lift_quotient_isomorphism(
    left_projection: ChainMap,
    right_projection: ChainMap,
    isomorphism: ChainMap,
    *,
    forward_lift: ChainMap | None = None,
    backward_lift: ChainMap | None = None,
) -> ChainMap:
    """Lift a specified quotient isomorphism to an automorphism of its source.

    Both projections must be epimorphisms from the same finite based complex
    C. Its positive-degree homology must vanish: over the supported fields
    this says C is projective in nonnegative chain complexes, a sum of S0 and
    disk complexes. In particular C=N(K[Delta^n]) satisfies the hypothesis.

    The result alpha satisfies right_projection * alpha = isomorphism *
    left_projection. Optional forward/backward chain lifts may be supplied;
    singular initial lifts are corrected using finite Fitting decompositions.
    Every supplied initial lift must commute with its quotient projection.
    """
    if not _same_complex(left_projection.source, right_projection.source):
        raise ValueError("Quotient projections must have the same based source complex.")
    source = left_projection.source
    _require_projective(source)
    _require_epimorphism(left_projection)
    _require_epimorphism(right_projection)
    if (not _same_complex(isomorphism.source, left_projection.target)
            or not _same_complex(isomorphism.target, right_projection.target)
            or not isomorphism.is_isomorphism()):
        raise ValueError("Supply a chain isomorphism between the two quotient targets.")
    desired = isomorphism.compose(left_projection)
    a = forward_lift if forward_lift is not None else _projective_lift(right_projection, desired)
    if (not _same_complex(a.source, source) or not _same_complex(a.target, source)
            or not _same_map(right_projection.compose(a), desired)):
        raise ValueError("The supplied forward lift does not lift the specified quotient map.")
    if a.is_isomorphism() and backward_lift is None:
        return a
    reverse = isomorphism.inverse().compose(right_projection)
    b = backward_lift if backward_lift is not None else _projective_lift(left_projection, reverse)
    if (not _same_complex(b.source, source) or not _same_complex(b.target, source)
            or not _same_map(left_projection.compose(b), reverse)):
        raise ValueError("The supplied backward lift does not lift the inverse quotient map.")
    if a.is_isomorphism():
        return a
    field = source.field
    exponent = max(source.dimensions, default=0)
    u, v = b.compose(a), a.compose(b)
    u_power = tuple(_power(field, value, exponent) for value in u.matrices)
    v_power = tuple(_power(field, value, exponent) for value in v.matrices)
    u_image = tuple(field.column_basis(value) for value in u_power)
    v_image = tuple(field.column_basis(value) for value in v_power)
    u_kernel = tuple(field.nullspace(value) for value in u_power)
    v_kernel = tuple(field.nullspace(value) for value in v_power)
    image_left, image_right = _subcomplex(source, u_image), _subcomplex(source, v_image)
    image_map = ChainMap(image_left, image_right, tuple(
        field.coordinates(v_image[q], field.matmul(a.matrix(q), u_image[q]))
        for q in range(len(source.dimensions))
    ))
    if not image_map.is_isomorphism():
        raise ArithmeticError("The stable Fitting images are not isomorphic.")
    correction = chain_isomorphism(_subcomplex(source, u_kernel), _subcomplex(source, v_kernel))
    matrices = []
    for q in range(len(source.dimensions)):
        basis = u_image[q].row_join(u_kernel[q])
        images = field.matmul(a.matrix(q), u_image[q]).row_join(
            field.matmul(v_kernel[q], correction.matrix(q)))
        matrices.append(field.matmul(images, field.coordinates(basis, sp.eye(source.dimension(q)))))
    result = ChainMap(source, source, tuple(matrices))
    if not result.is_isomorphism() or not _same_map(right_projection.compose(result), desired):
        raise ArithmeticError("The corrected quotient lift is not a commuting chain automorphism.")
    return result
