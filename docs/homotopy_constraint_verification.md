# `homotopy_constraint_verification.py` documentation

## Location
- Script: `experiments/homotopy_constraint_verification.py`

## Purpose
This script validates a combinatorial horn-index characterization by direct tensor-face computation.

For each tested tensor shape and horn index `j`, it compares two sets:
- `compute_missing_indices_dask(shape, j)` from `horn_map_reduce.py`
- the set of basis indices `m` such that `d_i(E_m) = 0` for every face `i != j`

## Algorithm
For each `(shape, horn_j)`:
1. Compute `n = min(shape) - 1` and skip if `horn_j` is out of range.
2. Compute `expected_indices = compute_missing_indices_dask(shape, horn_j)`.
3. Enumerate all multi-indices `m` in the shape.
4. For each horn face index `i` (`0..n`, excluding `horn_j`):
   - Build standard basis tensor `E_m`.
   - Compute `face(E_m, i)`.
   - Collect `m` where the face is the zero tensor.
5. Intersect these zero-face index sets over all `i != horn_j`.
6. Compare the intersection against `expected_indices` and report pass/fail.

## Outputs
The script prints:
- per-case headers with shape and horn index,
- counts from the combinatorial method and constraint method,
- per-case pass/fail result,
- final overall summary.

## Logging
- No file logger is configured.
- All reporting is written to standard output.

## Run
From repository root, either use editable install:

```powershell
.\.venv\Scripts\python.exe experiments\homotopy_constraint_verification.py
```

Or run with explicit source path:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe experiments\homotopy_constraint_verification.py
```