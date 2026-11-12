# Spectrum of the diagonal tensor boundary

This note gives the spectral formulas used by the finite filters in [adjoint_ops.py](../src/simplicial_tensors/adjoint_ops.py). The associated dense comparisons are in [test_adjoint_ops.py](../tests/test_adjoint_ops.py).

## Operators

Let $p\geq0$, let $k\geq1$ be the number of tensor axes, and let $c_a\geq0$ be fixed offsets with $\min_a c_a=0$. Set
$$
C_p=\mathbb R^{(p+1+c_1)\times\cdots\times(p+1+c_k)}.
$$
For $p\geq1$, $B_p:C_p\to C_{p-1}$ is the alternating sum of simultaneous index deletions. Use the unaugmented convention $B_0=0$. With entrywise Euclidean adjoints, put
$$
L_p^-=B_p^*B_p,\qquad L_p^+=B_{p+1}B_{p+1}^*,\qquad
\Delta_p=L_p^-+L_p^+.
$$

For a coordinate unit $e_m$, let $r(m)$ be the number of distinct active labels among its coordinates,
$$
r(m)=|\{m_a:m_a\leq p\}|.
$$
Then
$$
\Delta_p e_m=(r(m)+1)e_m,\qquad
\operatorname{spec}(\Delta_p)=\{2,\ldots,\min(k,p+1)+1\}.
$$
For $p\geq1$,
$$
\operatorname{spec}_{>0}(L_p^-)=\{2,\ldots,\min(k,p)+1\},
\qquad
\lambda_{\max}(L_p^-)=\min(k,p)+1.
$$
In degree zero, $L_0^-=0$.

## Pattern decomposition

Fix the equality pattern, the tail labels, and the order of the $r$ active labels in a coordinate unit. Let $g_0,\ldots,g_r$ count unused active labels before, between, and after them, and write $e_g$ for the corresponding coordinate unit. The diagonal change of sign
$$
e_g\longmapsto(-1)^{\sum_t t g_t}e_g
$$
identifies the boundary on this summand with the tensor-product differential of $K^{\otimes(r+1)}$. Here $K_j=\mathbb R$ for $j\geq0$, and $K_j\to K_{j-1}$ is the identity for odd $j$ and zero for even $j$. The Hodge operator of $K$ is the identity. The tensor-product signs cancel cross terms, so its Hodge operator is $(r+1)I$.

The chain identity gives $L_p^-L_p^+=L_p^+L_p^-=0$. On a full-Hodge eigenspace of eigenvalue $\lambda$, therefore $(L_p^-)^2=\lambda L_p^-$. Its lower-Hodge eigenvalues are zero and $\lambda$. The matching nonzero singular spectra in adjacent degrees give the listed lower-Hodge eigenvalues.

## Multiplicities

Write $c=(c_1,\ldots,c_k)$ and define
$$
E_r(c)=\sum_{j=0}^r(-1)^j\binom rj\prod_{a=1}^k(c_a+r-j).
$$
The full-Hodge eigenvalue $r+1$ has multiplicity
$$
M_{p,r}=\binom{p+1}{r}E_r(c).
$$
With all offsets zero this is $(p+1)_{\underline r}S(k,r)$, where $S(k,r)$ is a Stirling number of the second kind. The multiplicity of $r+1$ in $L_p^-$ is
$$
a_{p,r}=E_r(c)\sum_{\ell=0}^{\lfloor(p-r)/2\rfloor}
\binom{p-2\ell-1}{r-1},
\qquad 1\leq r\leq\min(k,p).
$$
The remaining eigenvalues of $L_p^-$ are zero.

## Filters in the implementation

Write $R=\min(k,p)$ and $L=L_p^-$. The orthogonal cycle projector is
$$
P_0=\prod_{\lambda=2}^{R+1}\left(I-\frac L\lambda\right).
$$
The empty product in degree zero is the identity. For matrices in degree at least two, $P_0=I-\frac56L+\frac16L^2$.

If $P(x)=\prod_{\lambda=2}^{R+1}(1-x/\lambda)$ and $g(x)=(1-P(x))/x$, then
$$
B_p^\dagger=g(L)B_p^*.
$$
Interpolation at $0,2,\ldots,R+1$ also evaluates the resolvent $(I+\mu L)^{-1}$.

The NumPy routines promote these evaluations to floating-point arithmetic. “Exact” describes the finite polynomial identity. Computed projector and pseudoinverse residuals are subject to roundoff. The current one-axis boundary implementation has a separate indexing defect, so vector filter calls require that defect to be repaired. The matrix and higher-axis tests exercise the supported examples.

This spectrum concerns the boundary operator and its adjoint. The full DSTM for the stated offsets is contractible. The spectral projectors are used for analysis, synthesis, and regularization as described in [boundary_homeostasis.md](boundary_homeostasis.md).
