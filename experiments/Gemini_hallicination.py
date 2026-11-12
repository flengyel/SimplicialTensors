"""Compute horn fillers for two fixed integer tensor products."""

import numpy as np

from simplicial_tensors.tensor_ops import (
    bdry,
    face,
    filler,
    horn,
    is_degen,
    kan_condition,
    n_hypergroupoid_comparison,
    n_hypergroupoid_conjecture,
)


def run_example(factors, missing_face, *, outer_horns):
    """Check one horn and full-module uniqueness for the selected horn family."""
    tensor = np.asarray(factors[0], dtype=object)
    for factor in factors[1:]:
        tensor = np.multiply.outer(tensor, np.asarray(factor, dtype=object))
    degree = min(tensor.shape) - 1
    selected_horn = horn(tensor, missing_face)
    selected_filler = filler(selected_horn, missing_face)
    result = {
        "shape": tensor.shape,
        "axes": tensor.ndim,
        "degree": degree,
        "horn_family": "outer and inner" if outer_horns else "inner",
        "shape_criterion": n_hypergroupoid_conjecture(tensor.shape),
        "unique_fillers": n_hypergroupoid_comparison(
            tensor, outer_horns=outer_horns, allow_degen=True
        ),
        "tensor_degenerate": is_degen(tensor),
        "boundary_degenerate": is_degen(bdry(tensor)),
        "missing_face": missing_face,
        "compatible": kan_condition(selected_horn, missing_face),
        "included_faces_equal": all(
            np.array_equal(face(selected_filler, i), selected_horn[i])
            for i in range(degree + 1) if i != missing_face
        ),
        "selected_filler_equal": np.array_equal(tensor, selected_filler),
    }
    print(f"shape={tensor.shape}, axes={tensor.ndim}, simplicial degree={degree}")
    if degree == 1:
        print("There are no inner horns in degree one; both outer horns are checked.")
    print(f"Checked horn family: {result['horn_family']}; unique fillers: {result['unique_fillers']}")
    print(f"Tensor degenerate: {result['tensor_degenerate']}; boundary degenerate: {result['boundary_degenerate']}")
    print(f"Omitted face={missing_face}; compatible: {result['compatible']}; "
          f"included faces agree: {result['included_faces_equal']}; "
          f"selected filler equals input: {result['selected_filler_equal']}")
    return result


def main():
    return [
        run_example(([0, 1, 2], [3, 4], [5, 6]), 0, outer_horns=True),
        run_example((range(6), range(3, 10), range(5, 11), range(7, 14)),
                    2, outer_horns=False),
    ]


if __name__ == "__main__":
    results = main()
    if not all(result["compatible"] and result["included_faces_equal"]
               and result["unique_fillers"] == result["shape_criterion"]
               for result in results):
        raise SystemExit(1)
