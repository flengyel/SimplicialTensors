"""Compare range tensors with the fillers selected from their horns."""

import argparse
from typing import Tuple

import numpy as np

from simplicial_tensors.tensor_ops import (
    bdry,
    filler,
    horn,
    is_degen,
    n_hypergroupoid_conjecture,
    range_tensor,
    reconstruct_range_tensor_from_horn,
)

DEFAULT_SHAPES = ((2, 2), (3, 3), (4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7))


def print_range_tensor(shape: Tuple[int, ...]) -> None:
    relation = "<" if n_hypergroupoid_conjecture(shape) else ">="
    print(f"shape={shape}, axes={len(shape)} {relation} simplicial degree={min(shape) - 1}")
    print(f"range tensor\n{range_tensor(shape)}\n")


def run_range_examples(shapes=DEFAULT_SHAPES, *, verbose=False):
    """Print each tensor and its own horn-reconstruction comparison result."""
    results = []
    for shape in shapes:
        print_range_tensor(shape)
        reconstructed = reconstruct_range_tensor_from_horn(
            shape, proceed_anyway=True, verbose=verbose
        )
        results.append(reconstructed)
        print(f"Selected horn fillers reproduce the range tensor of shape {shape}: {reconstructed}\n")
    return results


def print_matrix_example():
    matrix = np.array([[-6, 5, 7], [-5, 4, 8], [1, 9, 0]])
    selected_filler = filler(horn(matrix, 1), 1)
    print(f"Matrix equals the selected filler: {np.array_equal(matrix, selected_filler)}")
    print(f"Matrix is degenerate: {is_degen(matrix)}")
    boundary = bdry(matrix)
    print(f"Boundary\n{boundary}")
    print(f"Boundary is degenerate: {is_degen(boundary)}\n")


def _shape_argument(value):
    try:
        shape = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive axis lengths") from exc
    if not shape or any(length < 1 for length in shape):
        raise argparse.ArgumentTypeError("axis lengths must be positive")
    return shape


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape", type=_shape_argument, action="append", help="axis lengths, for example 3,3; may be repeated"
    )
    parser.add_argument("--verbose", action="store_true", help="print reconstruction diagnostics")
    args = parser.parse_args(argv)
    print("Examples of range tensors")
    run_range_examples(args.shape or DEFAULT_SHAPES, verbose=args.verbose)
    print_matrix_example()


if __name__ == "__main__":
    main()
