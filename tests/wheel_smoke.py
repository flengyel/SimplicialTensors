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
