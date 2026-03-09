# Verify Injective Face Map

## Source

```text
experiments/verify_injective_face_map.py
```

## Type

Python script

## Research Goal

Analyzes horn kernels, missing-index structure, and filler reconstruction behavior in concrete computational settings.

## Theoretical Anchors

- tex/horns.tex (support characterization and missing-index criterion).
- tex/normalization.tex (Moore filler map and split exact sequence).
- tex/nondegeneracy_lemma.tex (R_{p,j} ∩ D_p = {0} and horn decomposition).

## Typical Workflow

1. Configure shape/order/parameter choices directly in the script.
2. Run the script to evaluate identities or invariants associated with its target phenomenon.
3. Compare printed or saved outputs against the theorem-level expectations listed above.

## How To Run

```bash
python experiments/verify_injective_face_map.py
```

## Inputs

- Tensor shape data or graph/permutation objects defined in-script.
- The core simplicial_tensors implementation in src/simplicial_tensors.
- Optional scientific/python ecosystem dependencies required by the script.

## Outputs

- Horn compatibility diagnostics, filler comparisons, and kernel-support measurements.

## Interpretation Guidance

Use this script as computational evidence for manuscript-level claims, not as a standalone proof. Cross-check discrepancies against the cited TeX sections and their formal statements.

