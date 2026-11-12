# Codex handoff: SageMath verification in SagePod

Date: 2026-11-12

Continue work on [flengyel/SimplicialTensors](https://github.com/flengyel/SimplicialTensors).
Run the SageMath tests in [flengyel/sagepod](https://github.com/flengyel/sagepod),
repair demonstrated defects, update the affected documentation, and commit and
push the completed changes to remote `main`.

## Starting state

The implementation baseline is commit
`59947bc06a5bcdd9bb1c8246347af1a5377baed7`,
“Repair tensor computations and revise the manuscript.” Fetch current `main`
and retain subsequent commits. Do not reset the repository to this baseline.

That revision passed 320 tests in Python 3.12. The SageMath-specific test
module was skipped because SageMath was unavailable. Ruff, the selected Mypy
checks, Bandit, installation, documentation examples, and both ancillary
scripts passed. SageMath execution remains to be established.

The current paper is `tex/dstm_short.tex`, with inputs
`generated_section.tex` and `dendroidal_section.tex`. Its committed PDF has
ten pages. The full contraction proof, Quillen reference, generated-submodule
examples, graph representation, Grassmannian results, and tree-horn results
are retained.

Read `AGENTS.md` and `GUIDELINES.md` before editing. Reuse the shared library
implementations. Preserve unrelated local changes. Routine repairs and
fast-forward pushes to remote `main` are authorized. Do not force-push.

## Use the existing SagePod installation

Run Codex in the Linux shell, including the WSL2 Linux shell when applicable.
Inspect the installed SagePod checkout, running containers, image, and mounts
before changing the installation.

SagePod's current documented defaults are Sage 10.10, container `sagepod`,
and local image `localhost/sagepod-sagemath:10.10-pycryptosat`. The command
launcher is `/sage/sage` inside the container. These are discovery defaults;
record the actual versions and configuration.

```bash
podman ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
task_container=sagepod
podman inspect "$task_container" --format '{{json .Mounts}}'
podman exec "$task_container" /sage/sage -python -c \
  'from sage.env import SAGE_VERSION; import sys; print(SAGE_VERSION); print(sys.version)'
```

Reuse a running container. If the installation exists but the container is
stopped, use its `man-up.sh --no-follow` command. Follow its current README
only if setup or repair is needed. Do not rebuild, recreate, or stop a working
notebook session merely to run these tests.

The default mount maps host `~/Jupyter` to container
`/home/sage/notebooks`. Keep the test checkout in a directory visible through
the actual mount. A host path under `~/src` is not mounted by default.

One suitable checkout layout is:

```bash
task_repo="$HOME/Jupyter/SimplicialTensors"
task_container_repo=/home/sage/notebooks/SimplicialTensors
```

If that path does not exist, clone the repository there. Otherwise inspect
its Git status, branch, and remote before updating it. For a clean checkout
on `main`:

```bash
git -C "$task_repo" fetch origin
git -C "$task_repo" pull --ff-only origin main
git -C "$task_repo" merge-base --is-ancestor \
  59947bc06a5bcdd9bb1c8246347af1a5377baed7 HEAD
```

Do not overwrite an existing dirty checkout. An independent clone under the
mounted directory is suitable when the existing checkout must be preserved.
A linked worktree whose common Git directory is outside the mount needs that
directory to be accessible inside the container as well.

Use this host-shell function after adjusting the two container variables to
the observed configuration:

```bash
sage_repo() {
  podman exec -i \
    --workdir "$task_container_repo" \
    --env "PYTHONPATH=$task_container_repo/src" \
    "$task_container" /sage/sage -python "$@"
}
cd "$task_repo"
```

Inspect imports of `pytest`, `numpy`, `sympy`, and `scipy` in this interpreter.
Install only missing test dependencies in a writable Sage environment,
preserving Sage's supplied numerical packages. Host Python's packages do not
establish availability inside SagePod. An editable package installation is
optional when the source path above is used.

## Prove that the Sage backend is active

Run this gate before accepting any Sage test results:

```bash
sage_repo - <<'PY'
import sys
from sage.env import SAGE_VERSION
from sage.symbolic.expression import Expression
import simplicial_tensors.sagemath_compatible_tensor_ops as backend

print("Sage:", SAGE_VERSION)
print("Python:", sys.version)
print("Backend source:", backend.__file__)
assert backend.HAVE_SAGE, "SymPy fallback is active"
assert isinstance(backend.SymbolicTensor((2, 2)).tensor[0, 0], Expression)
PY
```

The backend currently catches import exceptions and falls back to SymPy.
A successful Python run with that fallback is insufficient for this task.
Diagnose any failure of the gate before proceeding.

Collect and run the dedicated module, saving its original failure output:

```bash
sage_repo -m pytest --collect-only -q tests/test_sagemath_compatible_tensor_ops.py
sage_repo -m pytest -o addopts= -vv -ra tests/test_sagemath_compatible_tensor_ops.py
```

The baseline module contains seven tests. Confirm execution, rather than
accepting a module-level skip.

## Correct the known test oracle and check native Sage scalars

The mixed simplicial identities in the skipped test contain two incorrect
right-hand sides. In `test_simplicial_identities`, the expression
`T.degen(j).face(i)` must be compared with:

- `T.face(i).degen(j - 1)` when `i < j`;
- `T` when `i == j` or `i == j + 1`;
- `T.face(i - 1).degen(j)` when `i > j + 1`.

These are the identities $d_i s_j=s_{j-1}d_i$ and
$d_i s_j=s_jd_{i-1}$ in the two nonidentity ranges. The baseline test instead
uses `T.face(j - 1).degen(i)` and `T.face(j).degen(i - 1)`.
For the exact matrix `np.arange(9, dtype=object).reshape(3, 3)`, the
cases `(i,j)=(0,2)` and `(2,0)` satisfy the corrected identities and fail
the original comparisons. Repair these assertions without weakening the
identities or altering the library to satisfy an incorrect oracle.

Run the remaining tests and repair any demonstrated backend defects. Add
bounded regressions using native Sage symbols and rationals. Existing shared
tests sometimes supply SymPy symbols to the Sage-compatible class.

Cover:

- Correction rank two for the entry expressions `x, 2*x, y, 3*y`, using
  Sage `var`; rational coefficients and the documented rejection of
  nonpolynomial expressions.
- Membership in a sum of degeneracy images, including a tensor outside
  each individual image; zero and nonzero tensors in degree zero.
- Nonuniqueness for the zero `3x3` tensor and uniqueness for the zero `4x4`
  tensor with `allow_degen=True`. Agreement with a selected filler alone
  does not prove uniqueness.
- Vector operations, rectangular shapes, and exact coefficients above
  $2^{53}$, with explicit exact conversions when comparing backends.

Use Sage `QQ` and `GF(p)` matrix ranks as independent references for small
absolute and relative tensor complexes. Compare with `generated_homology`
and `relative_horn_homology`, accounting for both boundary ranks and the
zero differential out of degree zero.

The current homology input contract accepts Python integers, `Fraction`,
and SymPy rationals. Arbitrary native Sage scalar types are not promised.
Convert them exactly for comparisons, or deliberately add and document
native-scalar support with regressions. Do not convert through floats.

## Run the integrated checks and Sage examples

After the focused repairs:

```bash
sage_repo -m pytest -o addopts= -q -ra
sage_repo experiments/sagemath_compatible_tensor_ops.py
sage_repo -m simplicial_tensors.notebooks.cocalc_compatible_tensor_ops
sage_repo experiments/homology_basis.py
sage_repo experiments/homology_basis.py --modulus 2
sage_repo experiments/normalized_moore_complex_homology.py --modulus 3
sage_repo experiments/sagemath_normalized_tensor_homology.py --modulus 3
sage_repo tex/anc/dstm_examples.py
sage_repo tex/anc/rank_shifted_homology.py
```

Execute the four current Python code cells of
`experiments/notebooks/cocalc_compatible_tensor_ops.ipynb`
in a shared Sage Python namespace. Record backend identity and completion.
Parsing notebook JSON alone does not establish Sage execution.

Run the Sage Petersen script with Sage's preparser:

```bash
podman exec -i --workdir "$task_container_repo" "$task_container" \
  /sage/sage experiments/notebooks/petersen_coset_enumeration.sage
```

Validate 30,240 distinct relabeled adjacency matrices, automorphism group
order 120, transversal rank counts `{4: 376, 6: 8496, 8: 21368}`, and
weighted counts `{4: 45120, 6: 1019520, 8: 2564160}`. Total weight $10!$
alone does not establish coverage. If the Sage/GAP action convention needs
repair, verify it against distinct matrices. The Python right-coset
implementation already has the correct convention; do not invert its
representatives merely to match another implementation.

For ordinary graph homology, the shifted tensor model must give Petersen
incidence rank 9 and Betti numbers $(1,6)$. Adjacency deletion-boundary ranks
describe a different calculation. The $E_{01}$-generated module has Betti
numbers $(0,1)$ although the full shape module is contractible.

Run the ordinary Python suite after shared-library changes, together with
the quality commands in `.github/workflows/ci.yml`. Optional PyTorch or Dask
tests may be skipped where those packages are absent; report those skips.
The dedicated Sage module must execute. Record actual counts in each
environment instead of copying the earlier total.

## Documentation, report, and publication

Update only documentation affected by demonstrated changes. Apply the
Mathematical Writing review and Došen–Petrić economy: definitions before use,
precise coefficient domains, concrete commands and results, professional
language, and no promotional or process prose in mathematical exposition.
Consult Knuth, Larrabee, and Roberts's *Mathematical Writing* when revising
mathematical prose. Preserve the ten-page limit if the manuscript changes;
rebuild its PDF and check references and layout.

Commit a concise validation report, for example
`docs/development/SAGEPOD_VALIDATION_2026-11-12.md`, containing:

- Tested source revision, SagePod revision or image ID, Sage/Python/package
  versions, backend gate, commands, pass/fail/skip counts, and relevant seeds.
- Defects found, exact repairs, and independent expected values.
- Notebook execution and Petersen coverage/rank results.
- Any unresolved failures, distinguished from unavailable optional packages.

Keep large raw logs outside the tracked source tree. Avoid recording
Jupyter tokens. Stage the task's files explicitly, inspect the diff, commit,
and push to remote `main`. Reconcile a concurrent remote update without
discarding it. Verify the published remote SHA and report it with the
validation outcome. Do not leave completed repairs only in the local checkout.
