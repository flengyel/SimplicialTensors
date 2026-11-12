# Rank-shifted tensors and graph chains

The module [rank_shifted.py](../../src/simplicial_tensors/rank_shifted.py) represents rational tensors sparsely in a fixed rank-shifted simplicial module. For fixed nonnegative shifts $c=(c_1,\ldots,c_k)$, degree $p$ has shape $(p+c_1+1,\ldots,p+c_k+1)$. Only labels $0,\ldots,p$ index faces and degeneracies. Tail coordinates add no simplicial maps.

`SparseTensor(p, c, entries)` stores a read-only map from coordinate tuples to exact `Fraction` coefficients. Omitted coordinates have coefficient zero. The class supports faces, degeneracies, positive-degree boundaries, normalization, degreewise outer products, and zero extension to larger shifts. Its degree-zero face and boundary methods raise `ValueError` because the zero target of the unaugmented differential has no tensor representation.

## Graph representation

`graph_embedding(vertex_count, edges, shift=None)` embeds the chain complex of a finite oriented graph into a three-axis shifted simplicial tensor module. Vertex labels are $1,\ldots,v$. Loops, repeated edges, isolated vertices, and the empty graph are allowed.

Number the edges separately from $1$ to $e$, writing $u_j\to w_j$ for edge $j$. Choose $r\ge\max(v,e)$, which is the default shift. The vertex and edge tensors are

$$
a_t=E_{(0,t,t)},\qquad
T_j=E_{(0,u_j+1,u_j+1)}+E_{(1,w_j+1,w_j+1)}+E_{(0,1,j+1)}.
$$

They satisfy $d_1T_j=a_{u_j}$ and $d_0T_j=a_{w_j}$. The normalized edge is $b_j=T_j-s_0a_{u_j}$, so

$$
d_0b_j=a_{w_j}-a_{u_j}.
$$

The marker coordinate $(0,1,j+1)$ makes the normalized edges linearly independent. All higher generators are degenerate. The normalized complex is therefore the graph's incidence complex in degrees 1 and 0. A tensor can also be viewed as a family of matrices indexed by its third coordinate.

The functions `graph_incidence_matrix` and `graph_chain_homology` in [graph_ops.py](../../src/simplicial_tensors/graph_ops.py) read the differential from these normalized tensor faces. The latter returns dimensions over $\mathbb Q$, or over $\mathbb F_p$ when a prime modulus is supplied.

~~~python
from simplicial_tensors.graph_ops import graph_chain_homology, graph_incidence_matrix
from simplicial_tensors.rank_shifted import graph_embedding

path = graph_embedding(3, [(1, 2), (2, 3)])
assert graph_incidence_matrix(path).tolist() == [[-1, 0], [1, -1], [0, 1]]
assert graph_chain_homology(path).homology_dimensions == (1, 0)
~~~

[The homology experiment](../experiments/homology_basis.md) computes paths, cycles, disconnected graphs, and the Petersen graph. The Petersen result has normalized dimensions $(10,15)$, incidence rank 9, and Betti numbers $(1,6)$.

## Adjacency calculations

The remaining `graph_ops` functions use a different graph representation. They accept a nonempty zero-diagonal symmetric zero-one adjacency matrix.

- `boundary_rank_distribution` counts ranks over $\mathbb F_2$ of its simplicial boundary matrix over all vertex orderings. Enumeration takes factorial time.
- `deletion_homology_mod2` computes homology of the deletion-only spans of induced adjacency matrices. It does not close under degeneracies.
- `adjacency_h1_relations` computes the nonnegative generator of the integral relation ideal on the normalized degree-one class. For a graph with an edge, the tensor-generated first homology is $\mathbb Z/(g)$, where $g$ is the returned integer. An edgeless graph generates the zero module.

These adjacency calculations differ from ordinary graph homology. A path on three vertices has ordinary Betti numbers $(1,0)$, whereas its adjacency-generated tensor module has zero rational homology and a nonzero first homology group over $\mathbb F_2$.
