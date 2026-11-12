"""Constructive chain embeddings and specified quotient-isomorphism lifts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
import sympy as sp

from simplicial_tensors.chain_complex import ChainComplex
from simplicial_tensors.chain_maps import (
    CapacityError, ChainMap, chain_embedding, chain_isomorphism,
    identity_chain_map, lift_quotient_isomorphism,
)


def interval(modulus=None):
    return ChainComplex((2, 1), (sp.zeros(0, 2), sp.Matrix([-1, 1])), modulus=modulus)


def interval_quotients(modulus=None):
    source = interval(modulus)
    left = ChainComplex((1, 1), (sp.zeros(0, 1), sp.Matrix([[-1]])), modulus=modulus)
    right = ChainComplex((1, 1), (sp.zeros(0, 1), sp.ones(1, 1)), modulus=modulus)
    pi_left = ChainMap(source, left, (sp.Matrix([[1, 0]]), sp.eye(1)))
    pi_right = ChainMap(source, right, (sp.Matrix([[0, 1]]), sp.eye(1)))
    phi = ChainMap(left, right, (sp.eye(1), sp.Matrix([[-1]])))
    return source, pi_left, pi_right, phi


def positive_tail_target(modulus=None):
    # N(X((1,1))) has three D1, two D2, and one S0 summand.
    d1 = sp.zeros(4, 5)
    d1[:3, :3] = sp.eye(3)
    d2 = sp.zeros(5, 2)
    d2[3:, :] = sp.eye(2)
    return ChainComplex((4, 5, 2), (sp.zeros(0, 4), d1, d2), modulus=modulus)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_chain_maps_compose_invert_and_snapshot_their_matrices(modulus):
    source = interval(modulus)
    mutable = sp.Matrix([[0, 1], [1, 0]])
    swap = ChainMap(source, source, (mutable, sp.Matrix([[-1]])))
    mutable[0, 0] = 7
    assert swap.matrices[0] == sp.Matrix([[0, 1], [1, 0]])
    assert swap.compose(swap.inverse()).matrices == identity_chain_map(source).matrices
    assert swap.inverse().compose(swap).matrices == identity_chain_map(source).matrices
    assert swap.is_injective() and swap.is_isomorphism()
    with pytest.raises(TypeError):
        swap.matrices[0][0, 0] = 7
    with pytest.raises(FrozenInstanceError):
        swap.matrices = ()
    assert swap.matrix(7).shape == (0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        swap.matrix(-1)


def test_invalid_chain_map_data_and_fields_are_rejected():
    source = interval()
    with pytest.raises(ValueError, match="identity"):
        ChainMap(source, source, (sp.eye(2), sp.zeros(1, 1)))
    with pytest.raises(ValueError, match="shape"):
        ChainMap(source, source, (sp.eye(1), sp.eye(1)))
    with pytest.raises(ValueError, match="Too many"):
        ChainMap(source, source, (sp.eye(2), sp.eye(1), sp.zeros(0, 0)))
    with pytest.raises(ValueError, match="same coefficient"):
        ChainMap(source, interval(2), ())
    zero = ChainMap(source, source, ())
    assert not zero.is_injective() and not zero.is_isomorphism()
    with pytest.raises(ValueError, match="isomorphism"):
        zero.inverse()


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_cycle_classes_can_embed_as_target_boundaries(modulus):
    circle = ChainComplex((0, 1), (sp.zeros(0, 0), sp.zeros(0, 1)), modulus=modulus)
    disk_two = ChainComplex((0, 1, 1), (sp.zeros(0, 0), sp.zeros(0, 1), sp.eye(1)), modulus=modulus)
    embedding = chain_embedding(circle, disk_two)
    assert embedding.is_injective()
    assert circle.homology().homology_dimensions == (0, 1)
    assert disk_two.homology().homology_dimensions == (0, 0, 0)
    assert embedding.matrix(1) == sp.eye(1)
    assert embedding.matrix(2).shape == (1, 0)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_positive_tail_target_requires_boundary_capacity_as_well_as_cycles(modulus):
    target = positive_tail_target(modulus)
    source = ChainComplex((4, 4), (sp.zeros(0, 4), sp.eye(4)), modulus=modulus)
    assert all(source.cycles(q).cols <= target.cycles(q).cols for q in range(3))
    with pytest.raises(CapacityError) as failure:
        chain_embedding(source, target)
    error = failure.value
    assert (error.degree, error.kind, error.required, error.available) == (0, "boundary", 4, 3)
    assert "required 4, available 3" in str(error)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_cycle_capacity_failure_is_explicit(modulus):
    source = ChainComplex((0, 1), (sp.zeros(0, 0), sp.zeros(0, 1)), modulus=modulus)
    target = ChainComplex((), (), modulus=modulus)
    with pytest.raises(CapacityError) as failure:
        chain_embedding(source, target)
    assert (failure.value.degree, failure.value.kind, failure.value.required,
            failure.value.available) == (1, "cycle", 1, 0)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_embedding_and_chain_isomorphism_use_arbitrary_based_coordinates(modulus):
    source = ChainComplex((2, 2, 1),
                          (sp.zeros(0, 2), sp.diag(1, 0), sp.Matrix([[0], [2]])),
                          modulus=modulus)
    field = source.field
    changes = (sp.Matrix([[1, 1], [0, 1]]), sp.Matrix([[1, 0], [1, 1]]), sp.eye(1))
    boundaries = [sp.zeros(0, 2)]
    for q in (1, 2):
        inverse = field.coordinates(changes[q], sp.eye(changes[q].rows))
        boundaries.append(field.matmul(changes[q - 1], field.matmul(source.differential(q), inverse)))
    target = ChainComplex(source.dimensions, tuple(boundaries), modulus=modulus)
    result = chain_isomorphism(source, target)
    assert result.is_isomorphism()
    assert result.inverse().compose(result).matrices == identity_chain_map(source).matrices
    with pytest.raises(ValueError, match="dimensions and differential ranks"):
        chain_isomorphism(source, interval(modulus))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_specified_interval_quotient_isomorphism_has_an_automorphism_lift(modulus):
    source, pi_left, pi_right, phi = interval_quotients(modulus)
    result = lift_quotient_isomorphism(pi_left, pi_right, phi)
    assert result.is_isomorphism()
    assert pi_right.compose(result).matrices == phi.compose(pi_left).matrices
    for q in range(len(source.dimensions)):
        kernel = source.field.nullspace(pi_left.matrix(q))
        assert not any(source.field.matmul(pi_right.matrix(q), source.field.matmul(result.matrix(q), kernel)))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_fitting_correction_repairs_two_singular_initial_lifts(modulus):
    source, pi_left, pi_right, phi = interval_quotients(modulus)
    forward = ChainMap(source, source, (sp.Matrix([[-1, 0], [1, 0]]), sp.Matrix([[-1]])))
    backward = ChainMap(source, source, (sp.Matrix([[0, 1], [0, -1]]), sp.Matrix([[-1]])))
    assert not forward.is_isomorphism() and not backward.is_isomorphism()
    assert pi_right.compose(forward).matrices == phi.compose(pi_left).matrices
    assert pi_left.compose(backward).matrices == phi.inverse().compose(pi_right).matrices
    corrected = lift_quotient_isomorphism(pi_left, pi_right, phi,
                                         forward_lift=forward, backward_lift=backward)
    assert corrected.is_isomorphism()
    assert pi_right.compose(corrected).matrices == phi.compose(pi_left).matrices
    stable = backward.compose(forward)
    for q in range(len(source.dimensions)):
        power = source.field.matmul(stable.matrix(q), stable.matrix(q))
        image = source.field.column_basis(power)
        difference = source.field.matrix(corrected.matrix(q) - forward.matrix(q))
        assert not any(source.field.matmul(difference, image))
        assert not any(source.field.matmul(pi_right.matrix(q), difference))
    assert corrected.inverse().compose(corrected).matrices == identity_chain_map(source).matrices


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_simplex_two_quotients_allow_positive_homology_in_the_quotients(modulus):
    d1 = sp.Matrix([[-1, -1, 0], [1, 0, -1], [0, 1, 1]])
    d2 = sp.Matrix([1, -1, 1])
    source = ChainComplex((3, 3, 1), (sp.zeros(0, 3), d1, d2), modulus=modulus)
    left = ChainComplex((1, 3, 1), (sp.zeros(0, 1), d1[:1, :], d2), modulus=modulus)
    right = ChainComplex((1, 3, 1), (sp.zeros(0, 1), d1[1:2, :], d2), modulus=modulus)
    pi_left = ChainMap(source, left, (sp.Matrix([[1, 0, 0]]), sp.eye(3), sp.eye(1)))
    pi_right = ChainMap(source, right, (sp.Matrix([[0, 1, 0]]), sp.eye(3), sp.eye(1)))
    phi = ChainMap(left, right, (sp.eye(1),
                                sp.Matrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
                                sp.Matrix([[-1]])))
    assert left.homology().homology_dimensions == (0, 1, 0)
    result = lift_quotient_isomorphism(pi_left, pi_right, phi)
    assert result.is_isomorphism()
    assert pi_right.compose(result).matrices == phi.compose(pi_left).matrices


def test_nonprojective_degreewise_free_source_is_rejected():
    source = ChainComplex((1, 2), (sp.zeros(0, 1), sp.Matrix([[1, 0]])))
    quotient = ChainComplex((0, 1), (sp.zeros(0, 0), sp.zeros(0, 1)))
    left = ChainMap(source, quotient, (sp.zeros(0, 1), sp.Matrix([[1, 0]])))
    right = ChainMap(source, quotient, (sp.zeros(0, 1), sp.Matrix([[0, 1]])))
    with pytest.raises(ValueError, match="projective"):
        lift_quotient_isomorphism(left, right, identity_chain_map(quotient))


def test_invalid_projection_isomorphism_and_initial_lifts_are_rejected():
    source, left, right, phi = interval_quotients()
    zero_projection = ChainMap(source, left.target, ())
    with pytest.raises(ValueError, match="surjective"):
        lift_quotient_isomorphism(zero_projection, right, phi)
    with pytest.raises(ValueError, match="isomorphism between"):
        lift_quotient_isomorphism(left, right, ChainMap(left.target, right.target, ()))
    with pytest.raises(ValueError, match="forward"):
        lift_quotient_isomorphism(left, right, phi, forward_lift=ChainMap(source, source, ()))
    identity = identity_chain_map(source)
    with pytest.raises(ValueError, match="backward"):
        lift_quotient_isomorphism(left, left, identity_chain_map(left.target),
                                  forward_lift=identity, backward_lift=ChainMap(source, source, ()))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_degree_zero_lift_retains_exact_large_rational_scalar(modulus):
    source = ChainComplex((1,), (sp.zeros(0, 1),), modulus=modulus)
    coefficient = Fraction(2**54 + 1, 5)
    phi = ChainMap(source, source, (sp.Matrix([[sp.Rational(coefficient)]]),))
    identity = identity_chain_map(source)
    result = lift_quotient_isomorphism(identity, identity, phi)
    assert result.matrix(0)[0, 0] == source.field.scalar(coefficient)
    assert result.inverse().compose(result).matrices == identity.matrices


def test_empty_complexes_and_trailing_zero_terms():
    empty = ChainComplex((), ())
    padded = ChainComplex((0, 0), (sp.zeros(0, 0), sp.zeros(0, 0)))
    result = chain_embedding(empty, padded)
    assert result.is_isomorphism()
    assert result.matrix(0).shape == (0, 0)
    identity = identity_chain_map(empty)
    assert lift_quotient_isomorphism(identity, identity, identity).is_isomorphism()


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_constructed_matrices_satisfy_native_sage_rank_and_chain_equations(modulus):
    sage = pytest.importorskip("sage.all")
    coefficient_field = sage.QQ if modulus is None else sage.GF(modulus)
    source, left, right, phi = interval_quotients(modulus)
    result = lift_quotient_isomorphism(left, right, phi)
    def native(values):
        return sage.matrix(coefficient_field, values.rows, values.cols,
                           [coefficient_field(int(v.p)) / coefficient_field(int(v.q)) for v in values])
    assert all(native(value).rank() == source.dimension(q)
               for q, value in enumerate(result.matrices))
    assert native(source.differential(1)) * native(result.matrix(1)) == (
        native(result.matrix(0)) * native(source.differential(1)))
    for q in range(len(source.dimensions)):
        assert native(right.matrix(q)) * native(result.matrix(q)) == (
            native(phi.matrix(q)) * native(left.matrix(q)))
