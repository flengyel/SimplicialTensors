# Finite dendroidal representable chains

[The experiment](../../experiments/dendroidal_representable_chains.py) computes six finite chain complexes: a binary corolla, a unary vertex above one input of a binary vertex, their boundary subcomplexes, and their relative boundary quotients. It uses the shared [exact chain-complex API](../../src/simplicial_tensors/chain_complex.py).

## Run

Install the package in the chosen Python environment, or set `PYTHONPATH=src`.

```sh
python experiments/dendroidal_representable_chains.py
python experiments/dendroidal_representable_chains.py --modulus 2
python experiments/dendroidal_representable_chains.py --modulus 3
```

The default coefficient field is $\mathbb Q$. The optional modulus must be prime. The experiment checks consecutive boundary products and the following dimensions and prints a JSON summary to standard output. It writes no files.

| Model | Chain dimensions | Boundary ranks | Betti numbers |
|---|---|---|---|
| Binary corolla | $(3,1)$ | $(0,1)$ | $(2,0)$ |
| Unary–binary tree | $(4,3,1)$ | $(0,2,1)$ | $(2,0,0)$ |
| Corolla boundary | $(3)$ | $(0)$ | $(3)$ |
| Unary–binary tree boundary | $(4,3)$ | $(0,2)$ | $(2,1)$ |
| Corolla relative to its boundary | $(0,1)$ | $(0,0)$ | $(0,1)$ |
| Unary–binary tree relative to its boundary | $(0,0,1)$ | $(0,0,0)$ | $(0,0,1)$ |

Dimensions are listed in ascending degrees, including the zero differential out of degree zero. These values hold over every field.

## Faces and representatives

For a one-vertex operation with root colour $r$ and input colours $a_1,\ldots,a_m$, its boundary is $\sum_i[a_i]-[r]$. In the two-vertex example, write $u:a\to e$, $v:(e,b)\to r$, and $w:(a,b)\to r$ for the composite. Root pruning, top pruning, and inner contraction give
$$
\partial[T]=[u]+[v]-[w],\qquad
\partial^2[T]=(a-e)+(e+b-r)-(a+b-r)=0.
$$
The full matrices and their bases are given in the [dendroidal research note](../research/dendroidal_homology.md#finite-representable-examples).

The relative models retain the top generator after quotienting out the proper faces. Their lift matrices express chosen representatives in the absolute bases. The chosen lift of the top generator has nonzero absolute boundary, which lies in the boundary subcomplex. Its image in the quotient is a cycle.

For the tree boundary, the cycle $u+v-w$ spans first homology. In the
short exact sequence of boundary, absolute, and relative chain complexes,
the connecting map sends the relative top class to $[u+v-w]$. This map
is an isomorphism from relative degree two to boundary degree one; the
relative class has no absolute cycle lift.

The signs follow Definition 3.1 of Bašić–Nikolaus, [*Homology of dendroidal sets*](https://arxiv.org/pdf/1509.00702). Corollaries 5.12–5.13 identify representable homology with the free group on its leaves in degree zero and the relative cell group in the vertex-count degree.

These are the chains of the two specified representables $\Omega[T]$, their boundaries, and their relative quotients. For the array presheaf $D(T)=A^{I(T)}$, the choice of homology construction and the normality issue are described in the research note.
