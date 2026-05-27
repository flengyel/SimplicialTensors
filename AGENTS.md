# AGENTS.md

## Repository rules

- Reusable tensor logic belongs under src/simplicial_tensors/.
- Experiments belong under experiments/.
- Do not duplicate bdry, degen, face, horn, or filler logic in experiments.
- For the neural-network boundary project, start with diagnostics before modifying training experiments.
- Do not describe s∂W as homological unless the diagnostic reports a nonzero relative/missing-index signal.
- Preserve superseded AI-generated experiments with explanatory headers rather than silently rewriting history.
- Run pytest -q after modifying library code.