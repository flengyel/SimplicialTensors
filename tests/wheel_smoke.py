"""Check an ordinary wheel installation from an isolated Python interpreter."""

from importlib import import_module, metadata, util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


NOTEBOOK_MODULES = (
    "cocalc_compatible_tensor_ops",
    "finite_difference_analysis",
    "finite_difference_analysis_horn_invariance",
    "homology_basis_dask",
    "homology_basis_with_degeneracy_check_dask",
    "horn_map_reduce",
)
RUNNABLE_MODULES = (
    "cocalc_compatible_tensor_ops",
    "homology_basis_dask",
    "homology_basis_with_degeneracy_check_dask",
)


def main() -> None:
    if not sys.flags.isolated:
        raise RuntimeError("Run this check with a fresh virtual environment's python -I.")
    checkout = Path(__file__).resolve().parents[1]
    prefix = Path(sys.prefix).resolve()
    distribution = metadata.distribution("SimplicialTensors")
    files = {str(path) for path in distribution.files or ()}
    expected = {
        f"simplicial_tensors/notebooks/{name}.py" for name in NOTEBOOK_MODULES
    } | {"simplicial_tensors/notebooks/__init__.py"}
    missing = expected - files
    if missing:
        raise RuntimeError(f"Wheel is missing supported modules: {sorted(missing)}")
    if any(name.endswith(".ipynb") for name in files):
        raise RuntimeError("Notebook artifacts are present in the installed wheel.")
    for name in NOTEBOOK_MODULES:
        qualified = f"simplicial_tensors.notebooks.{name}"
        spec = util.find_spec(qualified)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Installed module cannot be found: {qualified}")
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(prefix) or origin.is_relative_to(checkout):
            raise RuntimeError(f"Module was loaded outside the wheel environment: {origin}")

    import_module("simplicial_tensors.notebooks.cocalc_compatible_tensor_ops")
    from simplicial_tensors.chain_complex import ChainComplex
    from simplicial_tensors.chain_maps import chain_isomorphism, lift_quotient_isomorphism
    from simplicial_tensors.kernel_sequences import kernel_sequence, realize_kernel
    from simplicial_tensors.hypergraph_homology import embedded_chain_complex
    from simplicial_tensors.rank_shifted import circle_generator
    from simplicial_tensors.tensor_chain_complex import generated_chain_complex
    tensor_model = generated_chain_complex(circle_generator())
    assert isinstance(tensor_model, ChainComplex)
    assert tensor_model.homology().homology_dimensions == (0, 1)
    presentation = kernel_sequence(circle_generator())
    assert presentation.kernel.dimensions == (2, 0)
    reconstructed = realize_kernel(presentation.inclusion, (0, 0))
    assert kernel_sequence(reconstructed).quotient.homology().homology_dimensions == (0, 1)
    phi = chain_isomorphism(presentation.quotient, presentation.quotient)
    lifted = lift_quotient_isomorphism(presentation.projection, presentation.projection, phi)
    assert lifted.is_isomorphism()
    hypergraph_model = embedded_chain_complex([(0,), (1,), (0, 1)])
    assert hypergraph_model.homology().homology_dimensions == (1, 0)
    assert hypergraph_model.homology_representatives(0).cols == 1
    from dataclasses import replace
    from fractions import Fraction
    import numpy as np
    from simplicial_tensors.adjoint_ops import bdry_adjoint, boundary_pseudoinverse
    from simplicial_tensors.graph_ops import graph_chain_homology
    from simplicial_tensors.introspection_ops import boundary_analyze, boundary_synthesize
    from simplicial_tensors.rank_shifted import graph_embedding
    assert np.allclose(boundary_pseudoinverse(np.array([1], dtype=np.uint8), (2,)), [-0.5, 0.5])
    signal = np.random.default_rng(20261112).normal(size=(3, 4))
    visible = bdry_adjoint(signal, (4, 5))
    assert np.allclose(boundary_synthesize(*boundary_analyze(visible)), visible)
    graph = graph_embedding(2, [(1, 2)])
    graph = replace(graph, edge_tensors=(Fraction(1, 2) * graph.edge_tensors[0],))
    assert graph_chain_homology(graph, modulus=3).homology_dimensions == (1, 0)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="simplicial-wheel-run-") as directory:
        for name in RUNNABLE_MODULES:
            qualified = f"simplicial_tensors.notebooks.{name}"
            result = subprocess.run(
                [sys.executable, "-I", "-m", qualified],
                check=True, cwd=directory, env=environment,
                capture_output=True, text=True, timeout=120,
            )
            print(f"{qualified}: completed")
            print(result.stdout, end="")
    package_sources = sorted(
        name for name in files if name.startswith("simplicial_tensors/") and name.endswith(".py")
    )
    print(f"SimplicialTensors {distribution.version}: {len(package_sources)} Python source files")
    for name in package_sources:
        print(name)
    print("Wheel smoke check passed.")


if __name__ == "__main__":
    main()
