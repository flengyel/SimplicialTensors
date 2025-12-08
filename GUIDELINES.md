# SimplicialTensors Repository Guidelines

These guidelines describe how this repository is structured and how to extend it without creating duplicated, tangled, or “forked” code paths.

---

## 1. Directory layout (authoritative)

- `src/simplicial_tensors/`
  - The only place for reusable library code.
  - Contains core tensor and simplicial operations (for example `tensor_ops.py`, `symbolic_tensor_ops.py`, `cyclic_tensor_ops.py`).
  - Code here should be:
    - Shape-agnostic (no hard-coded 2×2, 3×3, etc.).
    - Written with clean function and class interfaces.
    - Importable without side effects (no heavy computations at import time).

- `experiments/`
  - Scripts that run actual experiments (graph examples, neural networks, cyclic operators, random tests, etc.).
  - Each script:
    - Imports functions from `simplicial_tensors`.
    - Documents in a short header comment what the experiment does.
    - May have a `main()` and an `if __name__ == "__main__":` block.

- `examples/`
  - Not for core code.
  - Prefer `experiments/` for any new runnable scripts.
  - Do not copy or mirror modules from `src/simplicial_tensors/` here.

- Other directories (`paper/`, `docs/`, `notebooks/`, etc.)
  - Hold TeX files, notes, slides, or notebooks.
  - Do not define the library API.

---

## 2. Core rules (no exceptions)

1. **Single source of truth**

   - Any algorithm, data structure, or operation used in more than one place must live in a module under:
     - `src/simplicial_tensors/`
   - Experiment scripts in `experiments/` must only call these via imports.

2. **No duplication**

   - Do not copy code from `src/simplicial_tensors/` into:
     - `experiments/`
     - `examples/`
     - any new directory.
   - If you are about to paste code that clearly already exists somewhere:
     - Stop, factor it into a function/class in `src/simplicial_tensors/`, and import it.

3. **No parallel library clones**

   - Do not create any second `simplicial_tensors` package in this or another subdirectory.
   - The only package named `simplicial_tensors` lives under `src/`.

4. **Boundary operator and DSTM**

   - Boundary, coboundary, horn, and DSTM-related operations are defined once, generically, in the library (for example in `tensor_ops.py`). fileciteturn0file1L1-L250
   - These operations must:
     - Work for arbitrary orders/tensor shapes given by a shape tuple.
     - Never be hard-coded to a particular matrix or tensor size.
   - Experiments that specialize to a particular graph, neural network, or tensor shape should live in `experiments/` and call the generic functions.

5. **Specialized operators**

   - More specialized but still reusable operations (for example, cyclic operators on tensors as in `cyclic_tensor_ops.py`) should either:
     - Live in a clearly named module under `src/simplicial_tensors/` if they are part of the library, or
     - Be wrapped and exercised in `experiments/` if they are exploratory. fileciteturn0file0L1-L80

---

## 3. Library code in `src/simplicial_tensors/`

Allowed:

- Pure functions and classes that:
  - Construct tensors (random, range, structured).
  - Implement face/degeneracy operations.
  - Implement boundary and coboundary maps.
  - Implement horn construction and filling.
  - Implement DSTM-style constructions and related utilities.
- Small helper functions for logging, random seeds, or shape utilities.

Strongly discouraged:

- Long-running experiments and demos.
- CLI parsing, plotting, or heavy I/O.
- Experiment-specific “main” workflows.

Optional:

- A small `if __name__ == "__main__":` block for quick internal tests is acceptable only if:
  - It uses tiny toy data.
  - It does not duplicate any `experiments/` script.
  - It can be removed without breaking anything.

---

## 4. Experiments (`experiments/`)

- Each file in `experiments/` is a standalone experiment.
- Typical structure:

  - A short header comment describing:
    - What tensor/graph/network is constructed.
    - Which library functions are being exercised.
    - What is printed or plotted.
  - Imports from `simplicial_tensors`.
  - A `main()` function that:
    - Builds the data.
    - Calls the library functions.
    - Prints or plots results.
  - A standard `if __name__ == "__main__":` guard that calls `main()`.

- It is perfectly fine to have **multiple** experiment scripts that all import from the same module (for example, several different experiments using `tensor_ops.py`); that is the desired pattern.
- Experiments must not reimplement or fork the same logic that already exists in the library.

---

## 5. AI / Codex usage rules

When using AI tools (Codex, GPT, etc.) in this repository:

1. Do not create new code under `examples/`.
2. Do not copy or paste code from `src/simplicial_tensors/` into any other directory.
3. To extend functionality:
   - Add reusable code to an appropriate module under `src/simplicial_tensors/`.
   - Create or update scripts under `experiments/` that import and call these functions.
4. Do not rename, move, or delete modules under `src/simplicial_tensors/` unless explicitly asked to do so in a human-written instruction.
5. If you need similar functionality in two places:
   - Refactor once into `src/simplicial_tensors/`.
   - Import it from both call sites.

---

## 6. Refactoring rule of thumb

Before adding or modifying code, ask:

- “Am I about to duplicate logic that already exists?”
  - If yes: factor it into a function/class in `src/simplicial_tensors/` and call that.
- “Is this a one-off experiment or a reusable building block?”
  - One-off → goes in `experiments/`.
  - Reusable → goes in `src/simplicial_tensors/`.

Following these rules keeps the repository usable for research and experimentation, without turning it into multiple inconsistent copies of the same code.
