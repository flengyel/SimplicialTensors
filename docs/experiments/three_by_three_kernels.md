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

The remaining 368 matrices include the all-ones matrix. Its kernel in degree zero is the plane with normal $(1,1,1)$, and its kernels in degrees one and two equal the full source spaces. Its generated submodule has Betti numbers $(1,0,0)$. [Appendix A](../../tex/generated_section.tex)
