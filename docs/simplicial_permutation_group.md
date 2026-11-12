# Deleting elements from permutation matrices

[The experiment](../experiments/simplicial_permutation_group.py) checks
that a rank-one correction followed by principal deletion maps
permutation matrices to permutation matrices. For a permutation matrix $P$ and standard basis
column $e_i$, it forms
$$
P+(Pe_i)(e_i^{\mathsf T}P)
$$
and deletes row and column $i$. The correction reconnects the
predecessor and successor of the deleted element in its permutation
cycle.

## Run

The script requires NumPy and has no command-line options.

```sh
python experiments/simplicial_permutation_group.py
```

The current constant is `n = 6` and the loop uses `range(n + 1)`.
It checks all $7!=5{,}040$ permutation matrices of size seven and
all seven deletion indices, giving 35,280 assertions. The source
comment mentioning $S_5$ and $S_4$ does not match these bounds.

Every input, corrected face, and successful check is printed.
The final line is `All tests passed.` if the loop completes.
Redirect standard output when preserving a complete run.

The corrected operation differs from the package's face, which only
deletes the common index. The script verifies that each output is a
permutation matrix. It does not test simplicial identities or
compatibility with group multiplication. Importing the file also
runs its test loop.
