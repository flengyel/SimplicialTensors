# Statistics of principal faces of permutation matrices

[The experiment](../experiments/permutation_statistics.py) samples one
permutation matrix per selected size and examines its principal faces.
It compares each face with

$$
\partial(P^2)-(\partial P)^2
$$

using cosine similarity. The powers are matrix products. Each face is
checked for the permutation property, nilpotence, idempotence, and
repetition among matrix powers. The calculations use the package signed
boundary and face operations with floating-point entries.

```sh
python -m pip install -e . matplotlib
MPLBACKEND=Agg python experiments/permutation_statistics.py --sizes 3 5 --seed 42 --max-powers 20 --output-dir plots/permutation-check
```

| Option | Default |
|---|---|
| `--sizes` | `3 5 7 11 13 17 19 20 21` |
| `--seed` | `42` |
| `--max-powers` | `220` |
| `--tolerance` | `1e-7` |
| `--verbose` | disabled |
| `--output-dir` | `plots` |

The seed selects one matrix at each size. Power-repetition checks use
`--max-powers` as their bound. Nilpotence checks use the input size as
their bound. Both, and the idempotence check, use the selected absolute
tolerance with zero relative tolerance.

The analysis prints property counts and returns one dictionary per face
with its index, deleted diagonal entry, property flags, repetition data,
similarity, and input size. `test_permutation_matrices` provides the
callable interface. `generate_visualizations` writes four files in the
output directory.

- `diagonal_vs_properties.png`
- `properties_by_size.png`
- `similarity_distribution.png`
- `properties_vs_similarity.png`

Counts describe the sampled matrices. A failed repetition check means no
repetition was found within its bound. Property counts can overlap.
