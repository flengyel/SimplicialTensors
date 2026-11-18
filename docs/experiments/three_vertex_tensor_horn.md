# An inner horn on a three-vertex tree

Give both tensor axes the edge set of the tree with operations $a\to e$, $(e,b)\to h$ and $h\to r$. Omit the contraction of $h$. The horn restriction is injective, and its cokernel is free of rank two. [Program](../../experiments/three_vertex_tensor_horn.py)

Run from the repository root with Python's standard library.

```sh
python experiments/three_vertex_tensor_horn.py
```

The program enumerates the single-edge and one-vertex subfaces of the three included faces. At each ordered pair of colours it identifies face values that agree by restriction to a common subface. There are 25 coordinates on the tree and 27 independent values on a compatible horn.

The top pruning and the contraction of $e$ both retain $b$ and $r$. Their common subfaces are contained in the single-edge face on $b$ or the unary face $h\to r$. Neither contains both $b$ and $r$. Consequently the two included faces may prescribe different values at $(b,r)$ and at $(r,b)$. Their two differences give the cokernel coordinates. A filler exists exactly when both differences vanish.
