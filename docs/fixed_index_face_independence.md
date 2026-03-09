# Fixed Index Face Independence

## Source

```text
experiments/fixed_index_face_independence.py
```

## Type

Python script

## Research Goal

Performs targeted computational checks of combinatorial identities and structural invariants on explicit tensor families.

## Relevant Results in TeX Sources

- tex/horns.tex (missing-index characterization and face-kernel basis).
- tex/combinatorics.tex (rank and finite-difference formulas).
- tex/normalization.tex (decomposition through Moore filler map).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script and record the checks it performs.
3. Compare printed or saved outputs against the expected behavior from the cited TeX sections.

## How To Run

```bash
python experiments/fixed_index_face_independence.py
```

## Inputs

- Inputs defined in the script (for example shapes, graphs, or permutations).
- Functions imported from src/simplicial_tensors.
- Optional dependencies required by the script.

## Outputs

- Consistency checks, discrepancy reports, and compact experiment summaries.

## Interpretation Guidance

Use this script to test the corresponding definitions and results. If output disagrees, check the cited TeX sections and the implementation.




