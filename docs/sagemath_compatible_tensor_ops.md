# `sagemath_compatible_tensor_ops.py` documentation

## Location
- Script: `experiments/sagemath_compatible_tensor_ops.py`

## Purpose
This script provides a symbolic tensor implementation (`SymbolicTensor`) for simplicial operations and runs symbolic horn/filler checks from `main()`.

It defines symbolic versions of:
- `face`, `degen`, `bdry`, `horn`, `filler`,
- symbolic degeneracy checks,
- horn filler uniqueness comparison,
- helper checks for symbol support in horns.

## Core operations
- `face(i)`: removes index `i` from each axis.
- `degen(k)`: duplicates index `k` on each axis.
- `bdry()`: alternating sum of faces.
- `horn(k)`: all faces with the `k`-th face replaced by a zero symbolic tensor.
- `filler(horn_list, k)`: Moore-style horn filler construction.
- `n_hypergroupoid_comparison(...)`: for each selected horn, verifies horn consistency and checks whether filler equals original tensor.

## What `main()` runs
1. Builds a symbolic `(3,3)` tensor and evaluates horn/filler comparison.
2. Runs `check_symbolic_corrections(...)` for horn index `1`.
3. Sweeps shapes `build_shape(k)` for `k=3..5` and all horn indices.
4. Runs additional checks on shape `(4,5,6)` and on `build_shape(d)` for `d=2..6`.

## Outputs
The script prints:
- conjecture prediction vs observed filler uniqueness,
- symbolic tensor/filler displays,
- per-shape horn check summaries,
- correction-symbol diagnostics from `check_symbolic_corrections(...)`.

## Logging
- No file logging is configured.
- All output is written to standard output.

## Run
From repository root:

```powershell
.\.venv\Scripts\python.exe experiments\sagemath_compatible_tensor_ops.py
```