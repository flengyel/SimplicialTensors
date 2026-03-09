# `horn_map_reduce.py` documentation

## Location
- Script: `experiments/horn_map_reduce.py`

## Purpose
This script computes and validates the set of "missing" multi-indices for a chosen horn face in a simplicial-style indexing setup.

It focuses on indices that contain every horn face index except the omitted one (`horn_j`), using Dask for filtered map/reduce-style processing.

## Dependencies
- `dask.bag`
- Python standard library: `itertools`, `functools.reduce`, `typing`

## Core Function
- `compute_missing_indices_dask(shape: Tuple[int, ...], horn_j: int) -> Set[Tuple[int, ...]]`

Behavior:
1. Let `order_k = len(shape)` and `dim_n = min(shape) - 1`.
2. Return `set()` immediately when:
   - `shape` is empty, or
   - `order_k < dim_n`.
3. Validate `horn_j` is in `[0, dim_n]`.
4. Build horn face indices `horn_faces = [k for k in range(dim_n + 1) if k != horn_j]`.
5. Generate all tensor indices via Cartesian product over `shape`.
6. Use Dask bag filtering to keep only indices containing all values in `horn_faces`.
7. Compute and return the final set.

## What `main()` does
1. Runs a small example: `shape=(2,2), horn_j=1`.
   - Compares against an expected set and asserts equality.
2. Runs a 3D example: `shape=(4,4,4), horn_j=0`.
   - Compares against `set(itertools.permutations([1,2,3]))` and asserts equality.
3. Runs a larger example: `shape=(10,10,10), horn_j=2`.
   - Prints count and performs a sanity check when non-empty.
4. Iterates over multiple shapes with fixed `horn_j=1` and prints count of missing indices for each.

## Output
The script prints:
- computed missing-index sets for the two asserted examples,
- assertion pass messages,
- count summary for the large example,
- count summary for a list of additional shapes.

Observed current run output includes these counts:
- `(2,2)` -> `3`
- `(3,3,3)` -> `12`
- `(3,5)` -> `2`
- `(3,3,5)` -> `16`
- `(3,3,3,5)` -> `74`
- `(3,3,3,3,5)` -> `280`
- `(3,4,5,6)` -> `144`
- `(3,3,3,3,3,5)` -> `962`
- `(3,3,3,3,3,3,5)` -> `3136`
- `(3,3,3,3,3,3,3,5)` -> `9914`

## Logging
This script does not configure file or structured logging.
All runtime reporting is printed to standard output.

## Run
From repository root:

```bash
.venv/Scripts/python.exe experiments/horn_map_reduce.py
```
