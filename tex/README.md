# Manuscript sources

The current manuscript is *Computations with simplicial tensor modules*.
The [compiled manuscript](dstm_short.pdf) has ten pages.
Build it from the repository root:

```sh
cd tex
latexmk -pdf -interaction=nonstopmode -halt-on-error dstm_short.tex
```

The driver is `dstm_short.tex`. Its inputs are `generated_section.tex`,
`dendroidal_section.tex` and `dstm_short.bib`. The `anc/` directory contains
the two example scripts and instructions for running them against the
package in this checkout.

To package the current source for arXiv, build the manuscript first and
run the following from `tex/`:

```sh
tar -czf dstm_arxiv_source.tar.gz dstm_short.tex generated_section.tex \
  dendroidal_section.tex dstm_short.bib dstm_short.bbl anc
```

The earlier driver `dstm.tex` and its section files correspond to
[arXiv:2512.10281v2](https://arxiv.org/abs/2512.10281v2).
