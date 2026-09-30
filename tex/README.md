# Manuscripts

The current short manuscript is **Horn kernels and normalization of diagonal
simplicial tensor modules**:

- [dstm_short.pdf](dstm_short.pdf): the recovered ten-page PDF, preserved unchanged.
- [dstm_short.tex](dstm_short.tex): the main source.
- [generated_section.tex](generated_section.tex): generated submodules and the
  appendix on kernel families, included by the main source.
- [dstm_short.bib](dstm_short.bib): the bibliography.

These files were recovered from version 6 of the short manuscript and its
source bundle, saved on September 29, 2026 at 03:11 UTC. The standalone PDF
and the PDF inside the source bundle have identical extracted text but
different bytes. The standalone PDF is the one preserved here; its SHA-256 is
`3516938dc724c2813b31228c70e2063cbd3354a4a5ef602d21a7a3057b429ac4`.

To build without overwriting that PDF, run from the repository root:

```bash
mkdir -p build/dstm-short
cd tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../build/dstm-short dstm_short.tex
```

## Section for later inclusion

[rank_shifted_homology.tex](rank_shifted_homology.tex) develops fixed rank
shifts, ambient homology, graph embeddings, the Petersen graph, and reduced
smash products. It is a separate section and is not included by either
manuscript driver. Its definitions and proofs accompany the
[sparse implementation and examples](../docs/rank_shifted_homology.md).

When integrating this section into the short article, place
`\input{rank_shifted_homology}` immediately before
`\input{generated_section}`. The latter file starts the appendix internally,
so inserting after it would make the new section part of the appendix.
The section uses the short article's existing theorem environments and
`Weibel` bibliography entry. Integration, renumbering, and edits to the
abstract and introduction are deferred.

## Earlier long manuscript

`dstm.tex`, `dstm.pdf`, and their original component files retain the earlier
long manuscript. They are not the current short draft. The rank-shifted
section is not incorporated into them.
