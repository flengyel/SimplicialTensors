"""Examples from the paper on diagonal simplicial tensor modules.

Run from the repository after ``python -m pip install -e .``:
    python tex/anc/dstm_examples.py

Matrix ranks are computed exactly over Q.
"""

from itertools import product
from math import prod

import numpy as np
from sympy import Matrix

from simplicial_tensors.kernel_sequences import kernel_sequence
from simplicial_tensors.rank_shifted import SparseTensor
from simplicial_tensors.tensor_ops import bdry, degen, face, filler, horn


def column(tensor):
    """Flatten an integer array into a column for exact linear algebra."""
    return Matrix(tensor.reshape(-1).tolist())


def shape_example():
    """Compute the ranks for shape (3, 5) in degrees zero through three."""
    shape = (3, 5)
    n = min(shape) - 1
    excesses = tuple(length - n - 1 for length in shape)
    assert n == 2 and excesses == (0, 2)
    ranks = []
    for p in range(4):
        degree_shape = SparseTensor.zero(p, excesses).shape
        if p == n:
            assert degree_shape == shape
        ranks.append(prod(degree_shape))
    assert ranks == [3, 8, 15, 24]
    print("Shape (3, 5): simplicial dimension", n, "and axis excesses", excesses)
    print("Ambient ranks in degrees 0 through 3:", ranks)


def vector_example():
    """One axis and simplicial degree one: deleting an entry leaves one."""
    tensor = np.array([3, 5], dtype=int)
    faces = [face(tensor, i) for i in range(2)]
    boundary = bdry(tensor)
    assert np.array_equal(faces[0], np.array([5]))
    assert np.array_equal(faces[1], np.array([3]))
    assert np.array_equal(boundary, np.array([2]))
    print("Vector faces:", [value.tolist() for value in faces])
    print("Vector boundary:", boundary.tolist())


def generated_homology_example():
    """E_01 generates H_1 = Q despite bounding in the ambient module."""
    tensor = np.array([[0, 1], [0, 0]], dtype=int)
    generators = [degen(tensor, 0), degen(tensor, 1)]
    primitive = np.zeros((3, 3), dtype=int)
    primitive[1, 2] = 1

    assert all(not np.any(face(tensor, i)) for i in range(2))
    assert np.array_equal(
        generators[0], np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    )
    assert np.array_equal(
        generators[1], np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    )

    # The two faces of tensor are zero, so Y_0 = 0 and Y_1 = Q tensor.
    # In degree two, Y_2 is spanned by its two degeneracies.
    y1 = column(tensor)
    y2 = Matrix.hstack(*(column(value) for value in generators))
    boundary1 = column(bdry(tensor))
    boundary2 = Matrix.hstack(*(column(bdry(value)) for value in generators))
    dim_y2 = y2.rank()
    boundary_rank = boundary2.rank()
    dim_h1 = y1.rank() - boundary1.rank() - boundary_rank
    primitive_in_y2 = Matrix.hstack(y2, column(primitive)).rank() == dim_y2
    primitive_has_boundary = np.array_equal(bdry(primitive), tensor)

    assert dim_y2 == 2
    assert boundary_rank == 0
    assert dim_h1 == 1
    assert primitive_has_boundary
    assert not primitive_in_y2
    print("dim Y_2:", dim_y2)
    print("rank of boundary on Y_2:", boundary_rank)
    print("dim H_1(Y):", dim_h1)
    print("Boundary of ambient primitive equals T:", primitive_has_boundary)
    print("Ambient primitive lies in Y_2:", primitive_in_y2)

    transposed = tensor.T
    for generator in (tensor, transposed):
        sequence = kernel_sequence(generator)
        assert sequence.source.dimensions == (2, 1)
        assert sequence.kernel.dimensions == (2, 0)
    assert Matrix.hstack(column(tensor), column(transposed)).rank() == 2
    print("E_01 and E_10 have L_0 = C_0 and L_1 = 0 and span distinct lines.")


def horn_example():
    """The missing-face-zero 2-horn has free entries in positions 12 and 21."""
    tensor = np.arange(1, 10, dtype=int).reshape(3, 3)
    prescribed = horn(tensor, 0)
    assert np.array_equal(prescribed[1], np.array([[1, 3], [7, 9]]))
    assert np.array_equal(prescribed[2], np.array([[1, 2], [4, 5]]))
    assert np.array_equal(face(prescribed[2], 1), face(prescribed[1], 1))

    computed = filler(prescribed, 0)
    filler_valid = np.array_equal(horn(computed, 0), prescribed)
    assert filler_valid
    sum_of_degeneracies = degen(prescribed[2], 1) + degen(prescribed[1] - prescribed[2], 0)
    np.testing.assert_array_equal(computed, sum_of_degeneracies)
    coordinates = (int(computed[1, 2]), int(computed[2, 1]))
    assert coordinates == (6, 8)

    # Compute the linear map B -> (d_1 B, d_2 B) on the nine matrix units.
    basis = []
    for row in range(3):
        for col in range(3):
            unit = np.zeros((3, 3), dtype=int)
            unit[row, col] = 1
            basis.append(unit)
    restriction = Matrix.hstack(
        *(Matrix.vstack(column(face(unit, 1)), column(face(unit, 2)))
          for unit in basis)
    )
    kernel_basis = Matrix.hstack(column(basis[5]), column(basis[7]))
    kernel_dimension = restriction.cols - restriction.rank()
    assert kernel_basis.rank() == kernel_dimension == 2
    assert restriction * kernel_basis == Matrix.zeros(8, 2)

    # All fillers are base + u E_12 + v E_21: a base point and the full kernel.
    base = np.array([[1, 2, 3], [4, 5, 0], [7, 0, 9]], dtype=int)
    assert np.array_equal(horn(base, 0), prescribed)
    reconstructed = base + computed[1, 2] * basis[5] + computed[2, 1] * basis[7]
    assert np.array_equal(computed, reconstructed)
    print("Horn faces d_1, d_2:", [prescribed[i].tolist() for i in (1, 2)])
    print("Horn kernel dimension:", kernel_dimension)
    print("Computed filler has the prescribed faces:", filler_valid)
    print("Coordinates (u, v) of the unique filler in D_2(X):", coordinates)
    print("All fillers: [[1, 2, 3], [4, 5, u], [7, v, 9]]")


def insert_zero(tensor):
    """Insert index zero on each axis and set the new entries to zero."""
    entries = {
        tuple(i + 1 for i in position): value
        for position, value in tensor.entries.items()
    }
    return SparseTensor(tensor.degree + 1, tensor.shifts, entries)


def matrix_units(p, shifts):
    rows, columns = SparseTensor.zero(p, shifts).shape
    for index in product(range(rows), range(columns)):
        yield index, SparseTensor.unit(p, shifts, index)


def is_normalized(unit):
    return all(not unit.face(i) for i in range(1, unit.degree + 1))


def square_matrix_example():
    """Compute the normalized basis and check the contraction on matrix units."""
    ranks = []
    images = {}
    for p in range(3):
        rank = 0
        for index, unit in matrix_units(p, (0, 0)):
            contracted = insert_zero(unit).boundary()
            if p > 0:
                contracted += insert_zero(unit.boundary())
            assert contracted == unit

            if is_normalized(unit):
                rank += 1
                if p > 0:
                    image = unit.face(0)
                    if image:
                        images[p, index] = dict(image.entries)
        ranks.append(rank)
    assert ranks == [1, 3, 2]
    assert images == {
        (1, (1, 1)): {(0, 0): 1},
        (2, (1, 2)): {(0, 1): 1},
        (2, (2, 1)): {(1, 0): 1},
    }
    print("Normalized ranks of square matrices in degrees 0 through 2:", ranks)
    for (p, index), image in images.items():
        print(f"Degree {p}: d_0 E{index} = E{next(iter(image))}")
    print("Verified the chain contraction on every matrix unit in degrees 0 through 2.")


def shifted_matrix_example():
    """Compute cycles and boundaries for the two-axis module with shifts (1, 1)."""
    cycles = []
    differential_ranks = []
    for p in range(4):
        basis = [unit for _, unit in matrix_units(p, (1, 1)) if is_normalized(unit)]
        # Each basis vector maps to zero or a matrix unit under d_0.
        images = {unit.face(0) for unit in basis} if p else set()
        rank = sum(bool(image) for image in images)
        differential_ranks.append(rank)
        cycles.append(len(basis) - rank)
    boundaries = differential_ranks[1:]
    assert cycles[:3] == [4, 2, 0] and boundaries == [3, 2, 0]
    print("Shifts (1, 1), degrees 0 through 2: cycles", cycles[:3], "boundaries", boundaries)


def main():
    shape_example()
    vector_example()
    generated_homology_example()
    horn_example()
    square_matrix_example()
    shifted_matrix_example()


if __name__ == "__main__":
    main()
