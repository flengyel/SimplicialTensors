# Symbolic boundaries of masked matrices

[Source](../experiments/random_zero_one_mask_cocycle_test.py) is an incomplete experiment comparing deletion boundaries after masking or relabeling symbolic entries. It imports `CochainComplex` from `cochain_complex_memoized_mod2`, which is absent from this repository and its package dependencies.

```sh
python experiments/random_zero_one_mask_cocycle_test.py
```

In the checked repository the command stops with `ModuleNotFoundError` at that import.

The intended default run samples 50 relabelings of a symbolic Petersen adjacency pattern and 20 pairs of random masks. It counts nonzero entries of each boundary difference and asks the missing helper whether the expressions are coboundaries. The samples have no configured seed.

The Petersen routine constructs masked `a_tensor` but computes `boundary_a` from the unmasked symbolic tensor `A`. Its comparison therefore combines masking with relabeling. The final “all tests passed” summary is unconditional. The helper and the comparison must be supplied and reconciled before this experiment supports a cohomology claim.
