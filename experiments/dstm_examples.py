"""Three computations from the diagonal simplicial tensor module note.

Run from the repository after ``python -m pip install -e .``:
    python experiments/dstm_examples.py

The script imports all simplicial operations from the library. Matrix ranks
are computed exactly over Q using SymPy; no floating-point rank test is used.
"""

import numpy as np
from sympy import Matrix

from simplicial_tensors.tensor_ops import bdry, degen, face, filler, horn


def column(tensor):
    """Flatten an integer array into a column for exact linear algebra."""
    return Matrix(tensor.reshape(-1).tolist())


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


def horn_example():
    """The missing-face-zero 2-horn has free entries in positions 12 and 21."""
    tensor = np.arange(1, 10, dtype=int).reshape(3, 3)
    prescribed = horn(tensor, 0)
    # horn() uses a floating-point zero placeholder. These particular data
    # are integers; restore their integer dtype before exact calculations.
    integer_horn = prescribed.astype(int)
    assert np.array_equal(prescribed, integer_horn)
    prescribed = integer_horn
    assert np.array_equal(prescribed[1], np.array([[1, 3], [7, 9]]))
    assert np.array_equal(prescribed[2], np.array([[1, 2], [4, 5]]))
    assert np.array_equal(face(prescribed[2], 1), face(prescribed[1], 1))

    computed = filler(prescribed, 0)
    filler_valid = np.array_equal(horn(computed, 0), prescribed)
    assert filler_valid

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
    print("All fillers: [[1, 2, 3], [4, 5, u], [7, v, 9]]")


def main():
    vector_example()
    generated_homology_example()
    horn_example()


if __name__ == "__main__":
    main()
