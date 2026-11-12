# Running repository commands

The commands in this repository run in a shell in a local clone.
In VS Code, use **Terminal → New Terminal** to open an integrated shell.
Other environments may expose a terminal differently.

Check the current directory and repository root.

```sh
pwd
git rev-parse --show-toplevel
git status --short
```

Change to the directory reported by Git before running the installation
or test commands. The checkout path depends on the machine and session.

If Git reports that the current directory is outside a repository,
locate an existing clone or create one.

```sh
git clone https://github.com/flengyel/SimplicialTensors.git
cd SimplicialTensors
```

Create and activate a Python environment, then install the package.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install pytest
python -m pytest -q
```

Use `.\.venv\Scripts\Activate.ps1` for activation in Windows PowerShell.
For import errors, check that the Python interpreter belongs to the
environment where the package was installed.

```sh
python -c "import sys, simplicial_tensors; print(sys.executable); print(simplicial_tensors.__file__)"
```
