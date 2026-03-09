# `sagemath_compatible_tensor_ops.py` documentation

## Locations
- Package module: `src/simplicial_tensors/sagemath_compatible_tensor_ops.py`
- Experiment entrypoint wrapper: `experiments/sagemath_compatible_tensor_ops.py`

## Purpose
This module provides a SageMath-oriented symbolic tensor implementation (`SymbolicTensor`) for simplicial operations.

The package module is the implementation source. The experiment file is a thin wrapper that imports and runs the package module's `main()`.

## Runtime backend behavior
- Preferred backend: SageMath symbolic API (`sage.all.var`, `sage.all.simplify`).
- Fallback backend: SymPy, used only when Sage is unavailable so the module remains importable in standard Python environments.

## Core operations
`SymbolicTensor` implements:
- tensor construction with symbolic entries (`range`, `zeros`, `ones`),
- `face(i)`, `degen(k)`, `bdry()`,
- `horn(k)`, `filler(horn_list, k)`,
- symbolic degeneracy check `is_degen()`,
- filler uniqueness check `n_hypergroupoid_comparison(...)`,
- arithmetic (`__add__`, `__sub__`) and symbolic helpers (`simplify`, `subs`).

## Uniqueness comparison algorithm
For each selected horn index:
1. Construct horn and candidate filler.
2. Verify non-missing horn faces are reproduced exactly.
3. Compare filler with original tensor entrywise.
4. If any entry differs for that horn, return `False`.
5. If all selected horns match exactly, return `True`.

This is a per-horn check, so each inner horn is validated independently.

## Outputs and logging
- No file logger is configured.
- Verbose diagnostics are printed to standard output when `verbose=True`.
- Non-verbose mode returns booleans/exceptions without extra logging.

## Run
From repository root:

```powershell
.\.venv\Scripts\python.exe experiments\sagemath_compatible_tensor_ops.py
```

Direct package import:

```powershell
.\.venv\Scripts\python.exe -c "import simplicial_tensors.sagemath_compatible_tensor_ops as m; print(m.HAVE_SAGE)"
```

SageMath execution (recommended for Sage backend):

```bash
sage -python -m pytest -q tests/test_sagemath_compatible_tensor_ops.py
```
## Recommended Direct Check (SageMath 10.7+)
Run this from repository root in a SageMath 10.7+ environment:

```bash
sage -python -m pytest -q tests/test_sagemath_compatible_tensor_ops.py
```

Optional backend confirmation:

```bash
sage -python -c "import simplicial_tensors.sagemath_compatible_tensor_ops as m; print('HAVE_SAGE=', m.HAVE_SAGE)"
```