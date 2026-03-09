# N Hypergroupoid Conjecture

## Source

```text
experiments/n_hypergroupoid_conjecture.py
```

## Type

Python script

## Research Goal

Tests the strict algebraic n-hypergroupoid criterion by sampling shapes/orders and comparing observed filler uniqueness to the k-versus-n threshold.

## Theoretical Anchors

- tex/combinatorics.tex (Hypergroupoid Classification theorem).
- tex/horns.tex (criterion R_{p,j} != 0 iff k >= p).
- tex/dichotomy.tex (filler dichotomy by order regime).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/n_hypergroupoid_conjecture.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Pass/fail traces of predicted versus observed uniqueness across sampled shapes.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

