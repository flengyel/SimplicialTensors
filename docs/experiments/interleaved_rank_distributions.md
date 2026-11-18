# Interleaved adjacency boundary ranks

For each order $\pi$ of a graph's $n$ vertices, insert an isolated vertex between each successive pair. Let $A_\pi$ be the adjacency matrix in the resulting order. Its mod-2 boundary $\partial A_\pi$ is the sum over $\mathbb F_2$ of its principal deletion faces. The experiment computes the distribution of the rank of $\partial A_\pi$ over all $n!$ orders.

From the repository root, run

```sh
PYTHONPATH=src python experiments/interleaved_rank_distributions.py
```

| Rank of $\partial A_\pi$ | Petersen | $4\times4$ rook | Shrikhande |
|---:|---:|---:|---:|
| 10 | 491,760 | 0 | 0 |
| 12 | 1,411,440 | 337,305,600 | 56,217,600 |
| 14 | 1,542,480 | 6,880,444,416 | 940,732,416 |
| 16 | 182,880 | 97,908,092,928 | 10,947,044,544 |
| 18 | 240 | 686,780,916,480 | 99,538,907,904 |
| 20 | 0 | 2,664,866,534,400 | 707,161,639,680 |
| 22 | 0 | 5,946,216,414,336 | 3,206,911,138,752 |
| 24 | 0 | 7,325,845,130,880 | 7,994,269,948,608 |
| 26 | 0 | 4,058,989,516,800 | 8,666,847,059,520 |
| 28 | 0 | 134,943,137,280 | 236,094,968,448 |
| 30 | 0 | 22,394,880 | 22,230,528 |
| **Total** | **3,628,800** | **20,922,789,888,000** | **20,922,789,888,000** |

All omitted ranks have frequency zero. Full rank is $2n-2$. Its frequency counts oriented Hamiltonian paths. For the original Petersen adjacency matrix, with no inserted vertices, the boundary ranks are $4,6,8$. [Petersen principal-deletion ranks](petersen_brd_right_cosets.md)

## Computation

Write $N$ for the strictly upper triangular part of the adjacency matrix of the original ordered graph. The interleaved boundary has rank twice the rank of $N$. For a prefix with vertex set $S$ and unused vertices $R$, partition the rows of $N$ indexed by $S$ as $(P\mid Q)$. Here $P$ has the prefix columns in their prescribed order, and $Q$ consists of the adjacency entries in the columns indexed by $R$. The computation retains the rank $t$ of $P$ and the subspace

$$
W=\{xQ : xP=0\}\subseteq\mathbb F_2^R.
$$

Append $v\in R$ to the prefix. The new column increases $t$ by one exactly when some $w\in W$ has $w_v=1$. The combinations of the old rows vanishing on the extended prefix are those with $w_v=0$. The new row vanishes on all prefix columns and restricts on $R'=R\setminus\{v\}$ to the neighbor vector $a_v$. Thus the new subspace is

$$
\{w|_{R'} : w\in W,\ w_v=0\} + \langle a_v\rangle.
$$

Starting from $(V,\{0\},0)$ with multiplicity one, the program sums the multiplicities for every choice of the next vertex. A graph automorphism preserves both ranks and permutes the remaining choices bijectively, so the program relabels $R$ and $W$ together to use a fixed representative of the orbit of $R$. With $R$ empty, the prefix rank is the rank of $N$. The program checks that the frequencies sum to $n!$ and that the full-rank frequency equals a separate count of Hamiltonian paths. [Program](../../experiments/interleaved_rank_distributions.py) · [Hamiltonian path computation](interleaved_graph_paths.md)
