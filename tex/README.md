# Manuscript sources

The current manuscript is *Computations with simplicial tensor modules*.
The [compiled manuscript](dstm_short.pdf) has fourteen pages.
The [arXiv replacement source archive](submissions/dstm_v3_arxiv_source.tar.gz)
contains the LaTeX source, the BibTeX database, the compiled bibliography,
and the two ancillary scripts with their instructions.
Build it from the repository root:

```sh
cd tex
latexmk -pdf -interaction=nonstopmode -halt-on-error dstm_short.tex
```

The driver is `dstm_short.tex`. Its inputs are `generated_section.tex`,
`dendroidal_section.tex` and `dstm_short.bib`.

From the repository root, install the package and run the four programs.

```sh
python -m pip install -e .
python tex/anc/dstm_examples.py
python tex/anc/rank_shifted_homology.py
python experiments/kernel_sequences.py
python experiments/interleaved_graph_paths.py
```

The two scripts in `tex/anc/` compute the matrix and horn examples and the
homology of the Petersen graph from its incidence complex. The
[kernel experiment](../docs/experiments/kernel_sequences.md) computes the
homology jump, higher-degree cycles, and lifts of quotient isomorphisms.
The [graph experiment](../docs/experiments/interleaved_graph_paths.md) counts
vertex orders attaining full boundary rank for the Petersen, rook, and
Shrikhande graphs. Both experiments are in the
[cited software revision](https://github.com/flengyel/SimplicialTensors/tree/68d68fd7f36d895fa2e11baba8727e0d2713e468/experiments).

To package the current source for arXiv, build the manuscript first and
run the following from `tex/`:

```sh
mkdir -p submissions
tar -czf submissions/dstm_v3_arxiv_source.tar.gz dstm_short.tex generated_section.tex \
  dendroidal_section.tex dstm_short.bib dstm_short.bbl anc
```

The driver `dstm.tex` and its section files correspond to
[arXiv:2512.10281v2](https://arxiv.org/abs/2512.10281v2).

Submission files are grouped in [submissions/](submissions/):

- [submission_7069953-20251210-2314.tar.gz](submissions/submission_7069953-20251210-2314.tar.gz)
- [submission_7069953-20251210-2342.tar.gz](submissions/submission_7069953-20251210-2342.tar.gz)
- [Submission 7069953 PDF, dated 11 December 2025](submissions/arXiv_submission.pdf)

The submission PDF is titled *Diagonal Simplicial Tensor Modules and
Algebraic n-Hypergroupoids*.
