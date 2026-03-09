# `verify_degenerate_preference.py` documentation

## Location
- Script: `experiments/verify_degenerate_preference.py`

## Purpose
This script tests whether `filler(horn(...))` returns the known degenerate tensor produced by `degen(...)`.

It runs a grid of cases over:
- a 2D base tensor (`2x2` standard basis matrix),
- a 3D base tensor (`3x3x3` standard basis tensor),
- every valid degeneracy index,
- every horn index of the degenerated tensor.

## Algorithm
For each `(base_tensor, degeneracy_index, horn_index)` case:
1. Build `T_degen = degen(base_tensor, degeneracy_index)`.
2. Build horn `H = horn(T_degen, horn_index)`.
3. Compute `T_fill = filler(H, horn_index)`.
4. Compare `T_fill` with `T_degen` using `np.array_equal`.

The script reports per-case pass/fail and then prints one overall pass/fail summary.

## Outputs
- Section headers for 2D and 3D test phases.
- Per-case status lines for each `(degeneracy, horn)` pair.
- Final summary line:
  - `PASS: Overall Result: ...` or
  - `FAIL: Overall Result: ...`

## Logging
- No file logging is configured.
- All output is written to standard output.

## Run
From repository root:

```powershell
.\.venv\Scripts\python.exe experiments\verify_degenerate_preference.py
```