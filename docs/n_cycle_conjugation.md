# Symbolic boundaries under cyclic relabeling

[Source](../experiments/n_cycle_conjugation.py) constructs an `n × n` matrix with independent symbolic entries, conjugates it by a cyclic permutation matrix, and compares the signed deletion boundaries.

```sh
python experiments/n_cycle_conjugation.py
```

The default run uses sizes 3 through 15 and prints each full difference. Its count is the number of nonzero symbolic entries. A `PASS` label means that the two boundary matrices agree entrywise.

A smaller call is `test_conjugation_by_n_cycle(3)`, imported from `experiments.n_cycle_conjugation`. Size 3 gives a false equality comparison, 4 nonzero entries, and reference value 4. Size 4 gives 8 nonzero entries and reference value 9. The variable `theoretical` is assigned `(n - 1)**2`, with no derivation or assertion relating it to the count.

The printed word “cocycles” labels entries in a boundary difference. This script does not compute cohomology classes or test whether these expressions are coboundaries.
