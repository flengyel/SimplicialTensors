# AGENTS.md

## Project purpose

This repository studies simplicial operations on matrices and hypermatrices, with an exploratory neural-network project based on boundary signals.

The neural-network project uses the boundary operator as an introspection signal:

- A weight tensor `W` has boundary `bdry(W)`.
- The identity `bdry(bdry(W)) == 0` means the boundary signal has no further boundary-level signal.
- Feedback into the original weight tensor is supplied by the adjoint boundary operator, not by an arbitrary lift.

## Repository structure

- Put reusable library code in `src/simplicial_tensors/`.
- Put runnable experiments in `experiments/`.
- Put explanatory notes in `docs/`.
- Put tests in `tests/`.
- Do not duplicate existing implementations of `face`, `degen`, `bdry`, `horn`, or `filler` inside experiments.

## Neural introspection rules

- Use the adjoint boundary map as the default return map from boundary tensors to weight tensors.
- For the diagonal face map `d_i`, where `d_i` deletes index `i` along every axis, implement `d_i^*` as zero-insertion into the positions avoiding index `i`.
- Implement the adjoint boundary as `bdry_adjoint(Y, original_shape) = sum_i (-1)^i d_i^*(Y)`.
- The default homeostatic feedback law is:
  `W <- W - eta * lambda * bdry_adjoint(bdry(W) - H, W.shape)`.
- The default differentiable regularizer is:
  `task_loss + 0.5 * lambda * ||bdry(W) - H||_F^2`.
- Do not describe an operator as an introspection mechanism unless it is tied to `bdry(W)`, `bdry(bdry(W)) == 0`, and a typed return map.
- Do not introduce untyped return maps, arbitrary lifts, or undocumented weight mutations.
- Do not mutate parameters between a forward pass and `loss.backward()` in PyTorch experiments.
- Match all control perturbations by relative norm when comparing feedback mechanisms.
- Treat improvements in task accuracy as empirical claims requiring controls, multiple seeds, and logged metrics.

## Implementation order

1. Implement and test adjoint boundary operators.
2. Add documentation for boundary homeostasis.
3. Add a small diagnostic script that reports boundary norms and feedback norms.
4. Add a controlled PyTorch experiment only after the adjoint operators and tests pass.

## Required tests for adjoint operators

- Verify `<face(W, i), Y> = <W, face_adjoint(Y, W.shape, i)>`.
- Verify `<bdry(W), Y> = <W, bdry_adjoint(Y, W.shape)>`.
- Verify `boundary_homeostasis_feedback(W)` has the same shape as `W`.
- Verify that a sufficiently small update in the direction `-bdry_adjoint(bdry(W), W.shape)` reduces `0.5 * ||bdry(W)||_F^2`.

Use deterministic random seeds in tests.

## Quality gates

- Run `pytest -q` after modifying library code.
- Add or update docs when public behavior changes.
- Keep experiments reproducible: fixed seeds, fixed data splits, logged hyperparameters.
- Prefer small, reviewable pull requests.
