"""Experiment entrypoint for the package SageMath-compatible symbolic tensor module."""

from simplicial_tensors.sagemath_compatible_tensor_ops import (
    HAVE_SAGE,
    SymbolicTensor,
    SimplicialException,
    n_hypergroupoid_conjecture,
    correction_rank,
    test_symbolic_n_hypergroupoid,
    check_symbolic_corrections,
    main as package_main,
)

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


def main() -> None:
    package_main()


if __name__ == "__main__":
    main()