# Statistics of principal faces of permutation matrices

[The experiment](../experiments/permutation_statistics.py) samples one
permutation matrix per selected size and examines its principal faces.
It compares each face with
$$
\partial(P^2)-(\partial P)^2
$$
using cosine similarity. Each face is checked for the permutation
property, nilpotence, idempotence, and repetition among matrix powers.

The calculations call the signed boundary and face functions from
`tensor_ops`. Entries are floating-point numbers. Nilpotence and
power-repetition tests use tolerances and finite power bounds.

## Current entry point

Direct execution raises `RecursionError`. The final
`def main(): main()` replaces the argument-parsing function.
This also prevents `--help` from reaching the parser.

The analysis functions can be called by loading the guarded module.

```sh
python -m pip install -e . matplotlib
python - <<'PYTHON'
import runpy

experiment = runpy.run_path("experiments/permutation_statistics.py")
results = experiment["test_permutation_matrices"](
    [3, 5], seed=42, max_powers=20, tolerance=1e-7
)
experiment["generate_visualizations"](results, "plots/permutation-check")
PYTHON
```

The analysis prints property counts and returns one dictionary per
face with the face index, deleted diagonal entry, property flags,
power-repetition data, similarity, and input size. The visualization
function writes `diagonal_vs_properties.png`,
`properties_by_size.png`, `similarity_distribution.png`, and
`properties_vs_similarity.png`.

## Declared command-line parameters

The earlier parser declares the following interface, currently blocked
by the recursive entry point.

| Option | Declared default |
|---|---|
| `--sizes` | `3 5 7 11 13 17 19 20 21` |
| `--seed` | `42` |
| `--max-powers` | `220` |
| `--tolerance` | `1e-7` |
| `--verbose` | disabled |
| `--output-dir` | `plots` |

That parser also fails to forward `--max-powers` and `--tolerance`
to `test_permutation_matrices`, whose defaults are 40 and `1e-7`.
An explicit call supplies both parameters.

Counts describe the sampled matrices. A failed repetition test means
no repetition was found within its bound. Several property tests can
hold for the same face, so their counts overlap.
