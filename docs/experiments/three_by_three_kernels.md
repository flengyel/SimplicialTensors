# Kernel planes of binary three-by-three matrices

The program computes the kernel sequence of every $3\times3$ matrix over $\mathbb F_2$, using the ordered vertex basis of the standard $2$-simplex. For each matrix $T$, the ranks of the three normalized maps are the dimensions of the quotient complex by the kernel sequence. [Experiment](../../experiments/three_by_three_kernels.py) · [Kernel-sequence API](../../src/simplicial_tensors/kernel_sequences.py)

Run from the repository root with the package and its dependencies installed. The command writes the counts and the all-ones example to standard output as JSON.

```sh
python experiments/three_by_three_kernels.py
```

Of the 512 labelled matrices, 144 attain the maximal ranks $(1,3,1)$ in degrees $0,1,2$ simultaneously. Their degree-one and degree-two kernels vanish. The degree-zero kernel is a plane with equation $ax+by+cz=0$. Its nonzero normal $(a,b,c)$ is unique over $\mathbb F_2$.

| Normal | Matrices on the maximal-rank locus |
|---|---:|
| $(0,0,1)$ | 24 |
| $(0,1,0)$ | 24 |
| $(0,1,1)$ | 24 |
| $(1,0,0)$ | 24 |
| $(1,0,1)$ | 24 |
| $(1,1,0)$ | 24 |
| $(1,1,1)$ | 0 |

Over $\mathbb F_q$, every plane except the one with normal $[1:1:1]$ occurs on the maximal-rank locus. Each of these $q^2+q$ planes has $q-1$ choices of diagonal representing its normal. For each diagonal the last row of the degree-one matrix is fixed and nonzero. The other two rows complete it to an ordered basis in $(q^3-q)(q^3-q^2)$ ways. At $q=2$ this gives the six counts of 24 in the table. [Proposition A.2 and the finite-field count](../../tex/generated_section.tex#L593)

The remaining 368 matrices include the all-ones matrix. Its kernel in degree zero is the plane with normal $(1,1,1)$, and its kernels in degrees one and two equal the full source spaces. Its generated submodule has Betti numbers $(1,0,0)$. [Appendix A](../../tex/generated_section.tex)
