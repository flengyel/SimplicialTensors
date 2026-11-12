# Embedded hypergraph homology

[hypergraph_homology.py](../../src/simplicial_tensors/hypergraph_homology.py)
computes embedded homology of finite hypergraphs over the rationals or a prime
field. It implements the infimum chain complex of
[Bressan, Li, Ren, and Wu](https://arxiv.org/html/1610.00890v5),
Proposition 2.1 and Section 3.2.

## Hyperedges and coefficient fields

A hypergraph $H$ is a finite collection of nonempty finite sets. Vertex labels
must be hashable and admit a common total order. Repeated hyperedges, and repeated
vertices within one hyperedge, are identified. The vertex set is the union of
the supplied hyperedges. A singleton hyperedge must be supplied explicitly;
an absent singleton is not inserted. The empty hypergraph has the empty complex.

Choose $F=\mathbb Q$, or $F=\mathbb F_p$ for a supplied prime modulus $p$.
The increasing vertex order orients every hyperedge. Let $K(H)$ consist of all
nonempty subsets of the hyperedges, and let $D_q\subseteq C_q(K(H);F)$ be the
span of the hyperedges with $q+1$ vertices, for $q\geq0$. The ambient simplicial differential $\partial_q$ is the alternating
sum of vertex deletions. Put $D_{-1}=0$ and $\partial_0=0$.

The infimum complex is
\[
I_q=D_q\cap\partial_q^{-1}(D_{q-1}).
\]
For $q\geq1$, if $x\in I_q$, then $\partial_qx\in D_{q-1}$ and
$\partial_{q-1}\partial_qx=0$, so $\partial_qx\in I_{q-1}$.
Its homology is the embedded homology of $H$.

## Computation and representatives

`embedded_homology(hyperedges, *, modulus=None)` returns chain dimensions,
boundary ranks, and homology dimensions. `embedded_chain_complex` accepts the
same arguments and returns a [based exact complex](chain_complex.md).
Its ambient row labels are the actual hyperedges, and its lift matrices express
chain basis vectors as linear combinations of those hyperedges.

Let $d_q$ be the number of $q$-hyperedges. Write $A_q$ for the ambient boundary
restricted to their columns and $M_q$ for its rows corresponding to absent
$(q-1)$-hyperedges. Then
\[
\dim I_q=d_q-\operatorname{rank}M_q,\qquad
\operatorname{rank}(\partial_q|_{I_q})=
\operatorname{rank}A_q-\operatorname{rank}M_q.
\]
The second equality follows from
$\ker A_q\subseteq\ker M_q$ and rank-nullity. The dimension-only routine uses
these ranks, with $A_0=M_0$ the zero matrix of shape $(0,d_0)$. The
chain-model routine solves the joint missing-face constraints. Both routines
assemble sparse boundaries using immediate facets and actual lower-degree
hyperedges.

For example, take all singleton hyperedges on $0,1,2,3$, the pairs
$01,12,23,03$, and the triples $012,023$, omitting the pair $02$. Neither triple
alone belongs to $I_2$, but their sum does:
\[
\partial([012]+[023])=[01]+[12]+[23]-[03].
\]
This is a linear constraint on combinations of hyperedges.

```python
from simplicial_tensors.hypergraph_homology import embedded_chain_complex

edges = [(0,), (1,), (2,), (3,), (0, 1), (1, 2), (2, 3), (0, 3),
         (0, 1, 2), (0, 2, 3)]
model = embedded_chain_complex(edges)
assert model.dimensions == (4, 4, 1)
assert model.homology().homology_dimensions == (1, 0, 0)
```

`associated_simplicial_complex(hyperedges)` explicitly constructs $K(H)$.
For an $m$-vertex hyperedge, this enumerates $2^m-1$ nonempty subsets. Passing
the closure to `embedded_homology` computes its usual simplicial homology.

Dropping absent faces from the ambient boundary need not give a differential.
For the singleton hyperedges $0,1,2$, the edge $01$, and the triple $012$, that
projection gives $\widehat\partial[012]=[01]$ and
$\widehat\partial^2[012]=[1]-[0]\ne0$. The infimum construction gives Betti numbers
$(2,0,0)$; the downward closure gives $(1,0,0)$.

The [example program](../experiments/hypergraph_homology.md) compares these models
and includes a field-dependent finite complex.
