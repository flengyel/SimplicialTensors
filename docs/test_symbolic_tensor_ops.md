# `test_symbolic_tensor_ops.py` documentation

## Location
- Test module: `tests/test_symbolic_tensor_ops.py`

## Purpose
This pytest module validates symbolic simplicial tensor behavior implemented in `simplicial_tensors.symbolic_tensor_ops`.

It checks:
- symbol naming and shape behavior,
- `face`, `degen`, `bdry`, `horn`, and `filler` consistency,
- `tensor_filler_difference_rank(...)`,
- `n_hypergroupoid_comparison(...)` behavior,
- simplicial identities and `bdry(bdry(T)) = 0` on randomized shapes.

## Test coverage summary
The module defines 14 tests:
1. variable naming
2. face/degen shape preservation
3. boundary construction
4. horn/filler face agreement
5. filler-difference rank type/value
6. conjecture/comparison compatibility check
7. no-inner-horns comparison case (`(2,2)`)
8. per-inner-horn comparison enforcement via monkeypatch
9. boundary-squared-zero randomized check
10. first simplicial identity
11. second simplicial identity
12. third simplicial identity
13. fourth simplicial identity
14. fifth simplicial identity

## Outputs
- Standard pytest progress and summary output.
- For normal passing runs, no custom file output is generated.

## Logging
- No file logging is configured by this test module.
- Output is standard pytest console output.

## Run
From repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_symbolic_tensor_ops.py
```

Run full repository tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```