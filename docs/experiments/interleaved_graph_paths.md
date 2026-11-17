# Hamiltonian paths and interleaved boundary ranks

The experiment counts the vertex orders of each graph that give a mod-two
boundary matrix of full rank. Given an order of the $n$ vertices, insert
one isolated vertex between each consecutive pair. The boundary of this
ordered adjacency matrix is the sum over $\mathbb F_2$ of the matrices
obtained by deleting one row and the corresponding column. It has
rank $2n-2$ exactly when the prescribed order
is a Hamiltonian path in the original graph.

## Run

From the repository root, with Python 3.10 or later and the package
dependencies installed, run

```sh
PYTHONPATH=src python experiments/interleaved_graph_paths.py
```

The script takes the three graphs as fixed inputs and prints the
following counts. Each path and its reverse count as two orders.

| Graph | Full boundary rank | Orders attaining full rank |
|---|---:|---:|
| Petersen | 18 | 240 |
| $4\times4$ rook | 30 | 22,394,880 |
| Shrikhande | 30 | 22,230,528 |

## Computation

For each possible starting vertex the script counts path extensions by
their current endpoint and the set of unused vertices. An empty set
contributes one completed path. Otherwise, the count is the sum over
unused neighbors of the endpoint, with each chosen neighbor removed
from the set. The script caches these counts and sums over starts.

For selected orders of each graph, the script also forms the interleaved
matrix and computes its boundary with `bdry`. It checks the matrix
entries and verifies that the boundary rank is twice the binary rank
of the strictly upper triangular part of the ordered adjacency matrix.
For both 16-vertex graphs the integer adjacency matrix satisfies
$A^2=4I+2J$, where $J$ is the all-ones matrix. They have the same
strongly regular parameters $(16,6,2,2)$ and adjacency eigenvalues
$6,2,-2$ with multiplicities $1,6,9$. Their different frequencies at
rank 30 distinguish the two interleaved boundary-rank distributions.

The [experiment without interleaved vertices](petersen_brd_right_cosets.md)
computes the rank distribution of the $9\times9$ boundaries of the
original Petersen adjacency matrices. The boundary counted here is
$18\times18$. [Source](../../experiments/interleaved_graph_paths.py)
