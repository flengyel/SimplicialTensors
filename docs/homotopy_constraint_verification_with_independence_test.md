# `homotopy_constraint_verification_with_independence_test.py` documentation

## Location
- Script: `experiments/homotopy_constraint_verification_with_independence_test.py`

## Purpose
This script performs two empirical checks over standard basis tensors:
1. linear independence of non-zero face images for each fixed face map,
2. horn-constraint equivalence between combinatorial missing indices and direct face-zero constraints.

## Algorithm
### Step 1: Linear independence check
For each tested `shape` and face index `i` in `0..dimen(shape_tensor)`:
1. Enumerate basis tensors `E_m`.
2. Compute `face(E_m, i)`.
3. Keep non-zero face images and flatten them to vectors.
4. Form a matrix of those vectors.
5. Compare matrix rank to number of vectors.
- Equal: reports independent.
- Not equal: reports dependent.

### Step 2: Constraint equivalence check
For each tested `shape` and horn index `j` in `0..dimen(shape_tensor)`:
1. Compute `expected_indices = compute_missing_indices_dask(shape, j)`.
2. Enumerate all basis indices `m`.
3. For every face index `i != j`, collect indices where `face(E_m, i)` is zero.
4. Intersect those sets across all `i != j`.
5. Compare the intersection with `expected_indices`.

## Outputs
The script prints:
- section headers for both steps,
- per-case counts and rank results in Step 1,
- per-case combinatorial/constraint counts in Step 2,
- pass/fail status per case,
- final summary lines for each step.

## Logging
- No file logger is configured.
- All reporting is written to standard output.

## Run
From repository root, either use editable install:

```powershell
.\.venv\Scripts\python.exe experiments\homotopy_constraint_verification_with_independence_test.py
```

Or run with explicit source path:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe experiments\homotopy_constraint_verification_with_independence_test.py
```