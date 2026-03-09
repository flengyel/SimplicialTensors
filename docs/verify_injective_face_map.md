# `verify_injective_face_map.py` documentation

## Location
- Script: `experiments/verify_injective_face_map.py`
- Log file: `logs/injectivity_verification.log`

## Purpose
This script checks an injectivity claim for simplicial face maps on standard basis tensors.

For each fixed shape and face index `i`, it tests whether the map
`m -> d_i(E_m)` is injective on indices `m` where `d_i(E_m)` is non-zero.

## Dependencies
The script imports:

```python
from simplicial_tensors.tensor_ops import face, dimen
```

## How It Works
For each test `shape` and each admissible `face_index`:
1. Enumerate all multi-indices `m` in the tensor shape.
2. Restrict to indices where `face_index not in m`.
3. Build the standard basis tensor `E_m`.
4. Compute `face(E_m, face_index)`.
5. Keep non-zero faces and record the corresponding source indices.
6. Detect duplicates by hashing flattened face tensors.

If any duplicate appears, that case fails injectivity.
If no duplicates appear, that case passes.

## Output
For each case, the script reports:
- number of non-zero faces generated,
- source indices used,
- duplicate count,
- PASS/FAIL for injectivity on that case.

At the end, it reports an overall PASS/FAIL summary across all tested shapes and face indices.

## Logging
- Console: concise progress and results.
- File: full run details at `logs/injectivity_verification.log`.

The log can be large because source-index lists are printed for each case.

## Run
From repository root:

```bash
PYTHONPATH=src .venv/Scripts/python.exe experiments/verify_injective_face_map.py
```
