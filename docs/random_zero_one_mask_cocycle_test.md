# Symbolic boundaries of masked matrices

[The experiment](../experiments/random_zero_one_mask_cocycle_test.py)
computes boundary differences for symbolic matrices with random masks
and for relabelings of a symbolic Petersen adjacency pattern. It uses
NumPy, SymPy, and the package boundary operation.

Given two source matrices $X$ and $Y$, it forms
$D=\partial X-\partial Y$ and checks the exact identities

$$
D=\partial(X-Y),\qquad \partial D=0.
$$

Thus each difference is a boundary with the displayed preimage. The
Petersen comparison applies the adjacency mask before computing either
boundary. Symbolic entries retain their matrix positions.

```sh
python experiments/random_zero_one_mask_cocycle_test.py --permutations 3 --masks 2 --seed 123
```

The defaults are 50 Petersen relabelings, 20 random mask pairs, seed 123,
and mask density 0.5. Calculations use $\mathbb F_2$ unless `--signed`
selects integer coefficients. The random-mask trials use the selected seed
plus one. All randomness is local to NumPy generators.

Each trial reports its pass status and the number of nonzero entries in
$D$. Petersen trials also report the permutation's cycle partition.
The summary counts successful checks, and a failed check gives exit
status 1. The script writes no files.

`boundary_difference_check` returns both source boundaries, their
difference, the nonzero-entry count, and Boolean boundary and cycle
checks. `test_random_zero_one_boundaries` and
`test_petersen_permutation_boundaries` return the trial records. Their
historical `*_cocycles` names remain as compatibility entry points.
The calculation concerns chain boundaries. It computes no cohomology
group or nonzero homology class.
