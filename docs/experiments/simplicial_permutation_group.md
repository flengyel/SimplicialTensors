# Deleting elements from permutation matrices

[The experiment](../../experiments/simplicial_permutation_group.py) checks that cycle deletion maps permutation matrices to permutation matrices. For a permutation matrix $P$ and standard basis column $e_i$, it forms
$$
P+(Pe_i)(e_i^{\mathsf T}P)
$$
and deletes row and column $i$. The correction reconnects the predecessor and successor of the deleted element in its permutation cycle. The matrix convention is $P_{j,\sigma(j)}=1$.

The script requires NumPy.

```sh
python experiments/simplicial_permutation_group.py
python experiments/simplicial_permutation_group.py --size 4 --verbose
```

`--size` is the number of elements and defaults to seven. The default checks all $7!=5{,}040$ permutation matrices and all seven deletion indices, giving 35,280 checks. The program raises an exception on the first failure and prints the number of completed checks on success. `--verbose` prints each input permutation, deletion index, and resulting matrix. Importing the module runs no checks.

This operation differs from the package's face, which deletes the common index without a correction. The experiment verifies that each output is a permutation matrix. It does not test compatibility with group multiplication.
