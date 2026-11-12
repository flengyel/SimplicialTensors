# Repository guidelines

## Layout

- `src/simplicial_tensors/` contains the reusable library.
- `tests/` contains unit, integration, documentation, and wheel-install checks.
- `experiments/` contains runnable studies and notebook documents.
- `experiments/results/<study>/` contains each study's data, figures, and logs.
  New runs write under its `runs/` subdirectory.
- `docs/api/`, `docs/experiments/`, `docs/research/`, and `docs/development/`
  separate API references, study guides, research notes, and repository workflows.
  [The documentation index](docs/README.md) provides navigation.
- `tex/` contains manuscript sources, PDFs, and ancillary computations;
  `tex/submissions/` contains submission packages and submission PDFs.
- `scripts/maintenance/` contains repository maintenance utilities.
- Importable notebook adapters and analysis helpers belong to
  `src/simplicial_tensors/notebooks/`; notebook documents belong to
  `experiments/notebooks/`.

## Library code

Implement reusable algorithms once in `src/simplicial_tensors/` and import
them from experiments. Keep the implementation generic in shape and
number of axes. Use explicit function and class interfaces.

Imports should avoid experiment execution, heavy computation, plotting,
or file output. Command-line parsing and experiment workflows belong
under `experiments/`. A small optional self-test in a library module may
use a main guard and tiny inputs.

New code must use the existing face, degeneracy, boundary, horn, and
filler implementations where applicable. Specialized reusable operators
belong in a clearly named library module. Experiment-specific use belongs
in the experiment that calls them.

## Experiments

Each experiment should state its purpose, required dependencies, input
parameters, output, and interpretation. Import reusable operations from
the package. Put execution in a `main()` function and a main guard when
practical.

Maintain a corresponding page in `docs/experiments/` or a linked API
reference for every tracked Python or Sage experiment, and keep
[the catalog](docs/experiments_catalog.md) consistent with the tracked files. Use fixed seeds and record parameters
for reproducible runs. Create output directories before writing files, and
keep default runs separate from checked-in result artifacts.

## Contributions

Read [AGENTS.md](AGENTS.md) for the neural boundary conventions and
required checks. Put new runnable experiments under `experiments/`.
Do not add new code under `examples/` or create another
`simplicial_tensors` package.

Do not copy library implementations into experiments or other
directories. If several scripts need the same operation, implement it
once in the library and import it.

Do not rename, move, or delete existing library modules without an
explicit human instruction. Before adding a new helper, check whether
an existing operation already supplies it.

Run `python -m pytest -q` after library changes. Verify documentation
commands and examples against the code they describe, and check the
diff before committing.
