# Repository guidelines

## Layout

- `src/simplicial_tensors/` contains the reusable library.
- `experiments/` contains experiment entry points, notebooks, and results.
- `docs/` contains documentation and research notes.
- `tex/` contains manuscript sources and ancillary scripts.
- Other source and output directories retain their existing roles.
  Their contents do not define the package API.

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

Maintain a corresponding page in `docs/` for every tracked Python or
Sage experiment and keep [the catalog](docs/experiments_catalog.md)
consistent with the tracked files. Use fixed seeds and record parameters
for reproducible runs.

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
