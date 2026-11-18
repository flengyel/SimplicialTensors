# Boundary ranks for affine orders of the rook and Shrikhande graphs

The experiment computes the mod-2 boundary of the adjacency matrix for
each of the 1,536 affine vertex orders of two Cayley graphs on
$(\mathbb Z/4)^2$. Write $e_1=(1,0)$ and $e_2=(0,1)$. In the rook graph,
two distinct vertices are adjacent if they agree in either coordinate.
In the Shrikhande graph, their difference must be one of
$\pm e_1,\pm e_2,\pm(e_1+e_2)$.

From the repository root, run

```sh
PYTHONPATH=src python experiments/affine_boundary_ranks.py
```

The program applies each map $x\mapsto Mx+t$ to the lexicographic order
of the 16 vertices. It enumerates the 96 two-by-two matrices modulo four
with odd determinant and the 16 translations. For each such order the
boundary is the sum over $\mathbb F_2$ of the 16 principal deletions of
the adjacency matrix. The experiment computes the rank of this
$15\times15$ matrix by binary elimination. Each affine map counts once.

| Boundary rank | $4\times4$ rook | Shrikhande |
|---:|---:|---:|
| 8 | 512 | 1,536 |
| 10 | 256 | 0 |
| 12 | 512 | 0 |
| 14 | 256 | 0 |
| Total | 1,536 | 1,536 |

The distribution over all vertex orders after insertion of isolated
vertices is a separate computation. [Affine-order program](../../experiments/affine_boundary_ranks.py) · [All orders after insertion](interleaved_rank_distributions.md)
