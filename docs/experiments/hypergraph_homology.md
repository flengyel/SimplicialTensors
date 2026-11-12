# Finite hypergraph homology examples

[The program](../../experiments/hypergraph_homology.py) compares embedded hypergraph
homology with simplicial homology of the downward closure. It uses the shared
[exact hypergraph API](../api/hypergraph_homology.md), with fixed finite inputs.

```sh
python experiments/hypergraph_homology.py
python experiments/hypergraph_homology.py --modulus 2
python experiments/hypergraph_homology.py --modulus 3 --case cancellation --representatives
```

The program prints JSON and writes no files. Each record gives the coefficient
field, infimum chain dimensions, boundary ranks, and the embedded and closure
Betti numbers.
`--case` selects one input. `--representatives` also gives hyperedge-coordinate
columns lifting a basis of each embedded homology group.

The triangle, filled triangle, missing-face, and cancellation examples specify
singleton hyperedges explicitly. The field-dependent example contains every face
of these ten triples on vertices $0,\ldots,5$:
\[
012,\ 013,\ 024,\ 035,\ 045,\ 125,\ 134,\ 145,\ 234,\ 235.
\]
Its chain dimensions are $(6,15,10)$. Its Betti numbers are $(1,0,0)$ over
$\mathbb Q$ and $\mathbb F_3$, and $(1,1,1)$ over $\mathbb F_2$.
The calculation records dimensions over fields; it does not identify an integral
homology group from those dimensions alone.

The cancellation example has infimum dimensions $(4,4,1)$ and Betti numbers
$(1,0,0)$ over all three fields. The missing-face example has embedded Betti
numbers $(2,0,0)$ and closure Betti numbers $(1,0,0)$.
