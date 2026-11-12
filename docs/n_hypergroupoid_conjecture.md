# Shape predicate and selected-filler comparison

[`tensor_ops.py`](../src/simplicial_tensors/tensor_ops.py) defines `n_hypergroupoid_conjecture(shape)` as `len(shape) < min(shape) - 1`. These are respectively the number of axes and the simplicial degree. In the manuscript's full module the horn kernel vanishes above the number of axes.

`n_hypergroupoid_comparison(a, outer_horns=False, verbose=False, allow_degen=False)` constructs horns of `a`, fills them with Moore's algorithm, and compares the selected fillers with `a`. It normally checks only inner horns. `outer_horns=True` includes endpoint horns. Unless `allow_degen=True`, a degenerate boundary raises `SimplicialException`.

A false return exhibits another filler for a checked horn. A true return means every selected filler equals `a`. That can hold for a degenerate input even when other fillers exist, so the Boolean alone does not certify uniqueness.

[The experiment](../experiments/n_hypergroupoid_conjecture.py) compares the checks on random arrays.

```sh
python experiments/n_hypergroupoid_conjecture.py
```

The default run requests 750 arrays of shape `(8, 10, 12)`, rejects degenerate boundaries, and performs three decreasing-degree checks. The rejection loops have no iteration limit. Dense degeneracy systems can be expensive.

A smaller call is

```python
from simplicial_tensors.tensor_ops import n_hypergroupoid_comparison, random_tensor

a = random_tensor((3, 3), low=1, high=10, seed=123)
print(n_hypergroupoid_comparison(a, allow_degen=True))
```

The experiment's `force_degeneracy` option changes the generated shape while its prediction uses the earlier `shape` variable. This mismatch affects forced-degeneracy comparisons.
