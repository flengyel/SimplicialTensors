# N Hypergroupoid Conjecture

## Source

```text
experiments/n_hypergroupoid_conjecture.py
```

## Type

Python script

## Research Goal

Tests the strict algebraic n-hypergroupoid criterion by sampling shapes/orders and comparing observed filler uniqueness to the k-versus-n threshold.

## Relevant Results in TeX Sources

- tex/combinatorics.tex (Hypergroupoid Classification theorem).
- tex/horns.tex (criterion R_{p,j} != 0 iff k >= p).
- tex/dichotomy.tex (filler dichotomy by order regime).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/n_hypergroupoid_conjecture.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Pass/fail traces of predicted versus observed uniqueness across sampled shapes.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




