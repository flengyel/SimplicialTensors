You are operating on the Git repository `SimplicialTensors` ([https://github.com/flengyel/SimplicialTensors](https://github.com/flengyel/SimplicialTensors)).
This is an experimental mathematics / computational algebra project implementing simplicial operations on matrices and tensors: face maps `d_i`, degeneracies `s_i`, boundary operators, and code for the diagonal simplicial tensor module (DSTM).

The previous automated refactor duplicated source files into multiple locations. That must be undone. There should be **one** canonical implementation of the core library, in a standard `src/` layout, and experiments should import from that one implementation.

Do **not** introduce any additional duplicated directory trees or copy entire modules into multiple locations.

---

### Step 0 – Repo sanity check

1. Assume the repo is already cloned locally. Work relative to the repo root.
2. Run:

   * `ls`
   * `ls src`
   * `ls src/simplicial_tensors`
   * `ls SimplicialTensors-Experiments`
   * `ls examples`
   * `ls experiments`
3. Read:

   * `pyproject.toml`
   * `README.md`
   * A representative module from `src/simplicial_tensors`, e.g. `tensor_ops.py` or whichever core module exists.
4. Identify any **secondary** or **historical** copies of the same logic (face maps, degeneracies, boundary operators, DSTM construction) outside `src/simplicial_tensors` (especially under `SimplicialTensors-Experiments`, `examples`, or `experiments`).

Goal of this step: understand where the canonical library lives (it should be `src/simplicial_tensors`) and where duplicated or legacy copies live.

---

### Step 1 – Choose a single canonical library

1. Treat `src/simplicial_tensors` as the **only** canonical implementation of:

   * Basic simplicial operations on tensors (face/degeneracy maps).
   * Boundary operators on tensors/matrices of arbitrary size.
   * Any DSTM-related construction that is “library-level” (reusable, not specific to one notebook or demo).
2. Do **not** move those modules out of `src/`, and do not create a second “library root” elsewhere.

If some newer or better implementation accidentally lives under `SimplicialTensors-Experiments` instead of `src/simplicial_tensors`, then:

* Move the improved implementation into `src/simplicial_tensors` and delete the duplicate copy from `SimplicialTensors-Experiments`.
* Adjust imports accordingly.

---

### Step 2 – Detect and remove duplicated modules

For each `.py` file outside `src/simplicial_tensors`:

1. Search for function and class names that also appear in the library:

   * Use something like `rg "def d_[0-9]+" -n` and `rg "boundary" -n`, `rg "tensor_ops" -n`, etc.
2. If a module in `SimplicialTensors-Experiments/` or `experiments/` or `examples/` defines functions/classes that **duplicate** ones in `src/simplicial_tensors` (same names and similar bodies), then:

   * Remove the duplicate function/class definitions from the experimental module.
   * Replace them by imports from `simplicial_tensors` (the installed package name) or from `src.simplicial_tensors` relative to the repo.

     * Example: `from simplicial_tensors.tensor_ops import d, s, boundary` (use the actual module names).
3. If there are entire duplicate files that are just copies of those under `src/simplicial_tensors`, delete the duplicates and update any imports to use the canonical module.

**Important rule:** after this step, there should be **no two files** that contain the same implementation of face maps, degeneracies, or boundary operators. Experiments may wrap or specialize them, but not re-implement them.

---

### Step 3 – Keep experiments, but make them thin wrappers

The top-level repo contains:

* `SimplicialTensors-Experiments/`
* `examples/`
* `experiments/`
* Notebooks under `docs/` or elsewhere.

These are allowed and encouraged, but:

1. They should import from `simplicial_tensors` (installed editable via `pip install -e .`), not define a second library.
2. For each experiment/demo script:

   * Ensure the top of the file imports the library:

     ```python
     from simplicial_tensors.tensor_ops import d, s, boundary  # adjust to actual module structure
     ```
   * Remove any local redefinitions that duplicate library functions.
   * Keep experiment-specific code (plots, randomization, neural-network glue, etc.) in these folders.

If `SimplicialTensors-Experiments` is essentially an older copy of the whole repo:

* Either:

  * Turn it into an “archive” directory by **removing** its internal `src`-like structure and keeping only genuinely different scripts/notebooks, or
  * Convert it to an `experiments/` sub-package that assumes `simplicial_tensors` is installed.

Do not leave a second `src` or package root inside `SimplicialTensors-Experiments`.

---

### Step 4 – Emphasize the boundary operator as a core primitive

The boundary operator is defined for matrices (and tensors) of arbitrary size and is intended to be used outside pure DSTM theory (e.g. for neural networks).

1. In the core library (`src/simplicial_tensors`):

   * Identify the main boundary operator implementation(s).
   * Add clear, high-level functions such as:

     ```python
     def boundary_matrix(A: np.ndarray, axis: int = 0) -> np.ndarray:
         """Apply the simplicial boundary operator to a matrix along the given axis."""
         ...
     ```

     (Adjust types / frameworks as in the existing code—NumPy, PyTorch, JAX, etc.)
   * Write docstrings that state clearly:

     * The operator works for any matrix size (no shape restriction beyond the simplicial indexing).
     * How the axes correspond to simplicial degrees.

2. Add at least one example script, e.g. under `examples/` or `experiments/nn/`:

   * Demonstrate applying the boundary operator to random weight matrices.
   * If the repo already uses PyTorch or similar, add a simple demonstration of using the boundary operator as a linear layer or regularizer in a tiny neural net example.
   * Keep this example **small** and focused; no large data dependencies.

3. Add tests in `tests/` that:

   * Verify the simplicial identities involving the boundary and face/degeneracy maps.
   * Check that the boundary operator behaves consistently for several matrix sizes and shapes.

---

### Step 5 – Packaging and imports

1. Ensure `pyproject.toml` (or `setup.cfg` if present) declares:

   * The package name `simplicial_tensors`.
   * `src` as the package root.
2. Run:

   * `pip install -e .`
   * `pytest`
3. Fix any import errors by:

   * Converting relative paths to canonical imports from `simplicial_tensors`.
   * Removing stale imports pointing into `SimplicialTensors-Experiments` or other legacy directories.

After this step, the intended workflow should be:

```bash
pip install -e .
python examples/tensor_ops.py
```

and experiments can import `simplicial_tensors` exactly as an external user would.

---

### Step 6 – Documentation touch-ups

1. Update `README.md` to reflect the cleaned structure:

   * Mention that the core library lives in `src/simplicial_tensors`.
   * Mention that experiments live in `examples/` and `experiments/`, and that they import from the library.
   * Add a short section “Neural network experiments” once the boundary operator example exists.

2. If you create `docs/FutureDirections.md` (the file above), reference it briefly in the README or in a short “Research notes” section.

