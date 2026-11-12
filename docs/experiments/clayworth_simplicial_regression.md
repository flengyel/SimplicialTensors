# Finite simplicial regression checks

[The script](../../experiments/clayworth_simplicial_regression.py) checks finite consequences of the Clayworth simplicial constructions against the package's face, degeneracy, horn, and filler operations.

Its tuple model enumerates states in `range(state_count)` through `max_dim`. For the strict orbit model, $\Phi(x)=\max(x-1,0)$ and adjacent tuple entries must be one update apart. Deleting an intermediate vertex or duplicating a non-fixed vertex can violate this condition. For example, $(2,1,0)$ is an orbit segment, whereas its middle face $(2,0)$ fails the one-step rule.

The Fano checks enumerate 16 closed objects, verify inner 2-horn filling in their inclusion poset, and exhibit an outer-horn obstruction. The all-tuples construction supplies a positive simplicial control, and the corresponding inner 2-horn test has a direct tuple filler.

```sh
python experiments/clayworth_simplicial_regression.py
python experiments/clayworth_simplicial_regression.py --json
python experiments/clayworth_simplicial_regression.py --max-dim 5 --state-count 5
```

The script inserts the repository `src` directory into `sys.path` and requires NumPy. Defaults are `--max-dim 4` and `--state-count 4`. Minimum permitted values are 2 and 3 respectively. It prints either a wrapped text table or a JSON list of diagnostic records. Exit status is zero when every expected outcome is observed.

The text labels `CHECKED`, `CLASSIFIES`, `FLAGS`, `REFUTES`, and `CONFIRMS` describe those expected outcomes. `ERROR` and `NO_CEX` identify a missing expected outcome. A `REFUTES` entry supplies a finite counterexample to the named closure or Kan assertion. The experiment's scope is these finite constructions. The proposed equivalence with a Grothendieck topos requires additional mathematical data and arguments.

## Source notice

The reviewed Clayworth material carried the following notice.

```text
© 2026 Logocentricity Inc.
U.S. Provisional Patent Application 63/961,154.
Released for academic citation, peer review, and scientific verification.
Paul Clayworth
Founder & CEO
Logocentricity Inc.
Capital Factory, Austin, TX
```
