# AGENTS.md

## Project purpose

This repository studies simplicial operations on matrices and hypermatrices.

## Repository structure

- Put reusable library code in `src/simplicial_tensors/`.
- Put runnable experiments in `experiments/`.
- Put API references in `docs/api/`, study guides in `docs/experiments/`,
  research notes in `docs/research/`, and repository workflows in `docs/development/`.
- Put notebook documents in `experiments/notebooks/` and each study's outputs
  under `experiments/results/<study>/`; new runs use its `runs/` subdirectory.
- Put tests in `tests/`.
- Do not duplicate existing implementations of `face`, `degen`, `bdry`, `horn`, or `filler` inside experiments.

## Boundary operators

- For the diagonal face `d_i`, implement its adjoint by inserting zeroes at index `i` on every axis.
- The boundary adjoint is the alternating sum of the face adjoints.
- For a tensor `W` and a boundary-shaped target `H`, the gradient of half the squared residual norm is `bdry_adjoint(bdry(W) - H, W.shape)`.
- To prescribe a boundary while preserving the cycle component, project the target into the boundary range and apply the boundary pseudoinverse. A fixed realizable target gives an idempotent update.

## Required tests for adjoint operators

- Verify `<face(W, i), Y> = <W, face_adjoint(Y, W.shape, i)>`.
- Verify `<bdry(W), Y> = <W, bdry_adjoint(Y, W.shape)>`.
- Verify `boundary_homeostasis_feedback(W)` has the same shape as `W`.
- Verify that a sufficiently small update in the direction `-bdry_adjoint(bdry(W), W.shape)` reduces `0.5 * ||bdry(W)||_F^2`.

Use deterministic random seeds in tests.

## Required tests for introspection operators

- Verify exact reconstruction from the cycle and boundary channels.
- Verify that synthesized controller outputs are projected into
  `range(bdry)` and pass the next-boundary metacheck.
- Verify exact attainment of a realizable target boundary.
- Verify preservation of the cycle channel.
- Verify idempotence for a fixed target or fixed projector controller.
- Verify the finite matrix implementation against a dense Moore--Penrose
  reference on small shapes.

## Quality gates

- Run `pytest -q` after modifying library code.
- Add or update docs when public behavior changes.
- Record input parameters and fix random seeds for reproducible experiments.
- Prefer small, reviewable pull requests.
