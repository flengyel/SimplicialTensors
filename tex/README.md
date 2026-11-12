# Manuscript sources

The current manuscript is *Computations with simplicial tensor modules*.
The [compiled manuscript](dstm_short.pdf) has fourteen pages.
Build it from the repository root:

```sh
cd tex
latexmk -pdf -interaction=nonstopmode -halt-on-error dstm_short.tex
```

The driver is `dstm_short.tex`. Its inputs are `generated_section.tex`,
`dendroidal_section.tex` and `dstm_short.bib`. The `anc/` directory contains
the two example scripts and instructions for running them against the
package in this checkout.

Section 6 proves quotient-isomorphism lifting and the criterion for realizing
a normalized kernel. It includes a matrix-family homology jump and sparse
coordinate generators in higher degrees. From an installed repository checkout,
run `python experiments/kernel_sequences.py` for their exact kernel,
quotient, realization, and lifting calculations; see
[the example guide](../docs/experiments/kernel_sequences.md).


To package the current source for arXiv, build the manuscript first and
run the following from `tex/`:

```sh
mkdir -p submissions
tar -czf submissions/dstm_arxiv_source.tar.gz dstm_short.tex generated_section.tex \
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
