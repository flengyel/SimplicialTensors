# Tensor restrictions on a two-vertex tree

For the rooted tree with vertices $a\to e$ and $(e,b)\to r$, omit the face that contracts $e$. In the two-axis tensor module with no additional indices, the horn restriction is surjective. The restriction of its kernel to the contracted face has cokernel of rank two. [Experiment](../../experiments/two_vertex_tensor_horn.py)

Run from the repository root with Python. The program uses the standard library and writes JSON to standard output.

```sh
python experiments/two_vertex_tensor_horn.py
```

The included faces have edge sets $\{a,e\}$ and $\{e,b,r\}$. A tensor on each face has one coordinate per ordered pair of edges. The two restrictions agree on $(e,e)$, the only coordinate on their common subface. The module of compatible pairs therefore has rank $2^2+3^2-1=12$.

| Restriction | Domain rank | Codomain rank | Image rank | Kernel rank | Cokernel rank |
|---|---:|---:|---:|---:|---:|
| Tree tensors to compatible horn tensors | 16 | 12 | 12 | 4 | 0 |
| Horn kernel to the boundary kernel of the contracted face | 4 | 6 | 4 | 0 | 2 |

The contracted corolla has edges $a,b,r$. A matrix lies in its boundary kernel if its three diagonal entries vanish. Of the six off-diagonal matrix units only $E_{(b,r)}$ and $E_{(r,b)}$ lie on an included face. Their classes give the reported basis of the cokernel. [Manuscript proof](../../tex/dendroidal_section.tex)
