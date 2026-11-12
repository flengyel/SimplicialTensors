# Horn kernels and filler comparisons

[`tensor_ops.py`](../../src/simplicial_tensors/tensor_ops.py) defines `n_hypergroupoid_conjecture(shape)` as `len(shape) < min(shape) - 1`. These are respectively the number of axes and the simplicial degree. For a nonzero coefficient ring, the full module's horn kernel vanishes exactly above the number of axes.

The function tests uniqueness in the current degree. The manuscript calls
a simplicial module an $r$-hypergroupoid when horn fillers are unique
in every degree greater than $r$.

`n_hypergroupoid_comparison(a, outer_horns=False, verbose=False, allow_degen=False)` validates Moore's selected fillers. A filler different from `a` exhibits nonuniqueness. If every selected filler equals `a`, the routine applies the horn-kernel criterion to decide uniqueness. Equality with a chosen filler alone does not determine the return value.

The default checks inner horns. `outer_horns=True` includes endpoint horns. Unless `allow_degen=True`, a degenerate boundary raises `SimplicialException`. With no selected horns, as for the inner horns of a degree-one tensor, the result is `True` vacuously. Use outer horns to test positive-degree tensors in that case.

## Experiment

[experiments/n_hypergroupoid_conjecture.py](../../experiments/n_hypergroupoid_conjecture.py) uses reproducible integer samples. Its defaults are ten tests, shape `(3, 3)`, seed 123, and at most 100 attempts per accepted sample. It compares fillers with `allow_degen=True`. `--skip-degeneracies` requires a nondegenerate boundary unless forced degeneracy is requested.

```sh
python experiments/n_hypergroupoid_conjecture.py --outer-horns
python experiments/n_hypergroupoid_conjecture.py --shape 3,3 --force-degeneracy --outer-horns
```

The prediction uses each tensor's shape after any forced degeneracy. Other options are `--random-shapes`, `--maxdim`, `--max-attempts`, `--verify-descent`, and `--verbose`. The optional decreasing-degree checks use the supplied `--shape`, whose default is `(3, 3)`. They require both a nondegenerate tensor and a nondegenerate boundary, apply three degeneracies, and descend through faces using the same bounded rejection policy. The library helper `verify_unique_down_to_order()` has default shape `(4, 4, 4)`. A requested nondegeneracy filter raises `ValueError` immediately when the simplicial degree exceeds the number of axes, since every tensor in that degree is degenerate. Otherwise, exhausting the attempt limit raises `RuntimeError`.

A library call is

```python
import numpy as np
from simplicial_tensors.tensor_ops import n_hypergroupoid_comparison

assert not n_hypergroupoid_comparison(np.zeros((3, 3)), allow_degen=True)
assert n_hypergroupoid_comparison(np.zeros((4, 4)), allow_degen=True)
```

Both tensors equal their selected Moore fillers. The first shape has a nonzero horn kernel, while the second has zero horn kernel.
