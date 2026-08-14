# Candidate exact spectrum theorem for the diagonal tensor boundary

This note records a result derived during the neural-boundary study.  It was
not found in the literature search and should receive independent proof
review before being presented as new mathematics.  Dense calculations and
matrix-free identities are tested in `tests/test_adjoint_ops.py`.

## Setup

Fix degree (pge0), tensor order (k), and offsets (c_age0) with
(min_a c_a=0).  Let

\[
C_p=\mathbb R^{(p+1+c_1)\times\cdots\times(p+1+c_k)}.
\]

Let (B_p:C_p\to C_{p-1}) be the diagonal boundary that deletes the same
common label (i\in\{0,\ldots,p\}) on every axis, with alternating signs.
Define

\[
L_p^-=B_p^*B_p,
\qquad
L_p^+=B_{p+1}B_{p+1}^*,
\qquad
\Delta_p=L_p^-+L_p^+.
\]

All adjoints use the entrywise Euclidean inner product.

## Theorem

For a standard basis tensor (e_m), let

\[
r(m)=\left|\{m_a:m_a\le p\}\right|,
\]

the number of distinct common labels occurring in its multi-index.  Then

\[
\Delta_pe_m=(r(m)+1)e_m.
\]

Thus the full Hodge operator is diagonal in the entry basis, with spectrum

\[
\operatorname{spec}(\Delta_p)
=\{2,3,\ldots,\min(k,p+1)+1\}.
\]

The lower Hodge operator has exact nonzero spectrum

\[
\operatorname{spec}_{>0}(L_p^-)
=\{2,3,\ldots,\min(k,p)+1\}.
\]

In particular,

\[
\lambda_{\max}(L_p^-)=\min(k,p)+1.
\]

## Proof sketch

Fix the equality pattern, private labels, and ordering pattern of a basis
multi-index using (r) common labels.  Encode the positions of the selected
common labels by the (r+1) gap lengths
(g=(g_0,\ldots,g_r)).  After the diagonal sign twist

\[
e_g\longmapsto(-1)^{\sum_t t g_t}e_g,
\]

the restricted diagonal boundary is the total tensor-product differential on
(K^{\otimes(r+1)}), where (K_j=\mathbb R) and the differential
(K_j\to K_{j-1}) is (1) for odd (j) and (0) for even (j).  The Hodge
operator of (K) is the identity in every degree.  Koszul signs cancel the
cross terms in the tensor product, so the full Hodge operator on this pattern
summand is ((r+1)I).  Summing the invariant pattern summands gives the stated
entrywise formula.

Since (B_pB_{p+1}=0),

\[
L_p^-L_p^+=L_p^+L_p^-=0.
\]

On a full-Hodge eigenspace with eigenvalue (lambda), this implies

\[
(L_p^-)^2=\lambda L_p^-,
\]

so the lower eigenvalues on that subspace are only (0) and (lambda).
Matching the nonzero singular spectra of (B_{p+1}B_{p+1}^*) and
(B_{p+1}^*B_{p+1}), together with the degree-zero base case, yields every
integer (2,\ldots,\min(k,p)+1) with positive multiplicity.

## Multiplicities

Define the fixed-pattern count

\[
E_r(c)=\sum_{j=0}^{r}(-1)^j{r\choose j}
\prod_{a=1}^{k}(c_a+r-j).
\]

The full eigenvalue (r+1) has multiplicity

\[
M_{p,r}={p+1\choose r}E_r(c).
\]

For constant shapes (c=0), this becomes

\[
M_{p,r}=(p+1)_{\underline r}S(k,r),
\]

where (S(k,r)) is a Stirling number of the second kind.

The multiplicity of (r+1) in the lower operator is

\[
a_{p,r}=E_r(c)
\sum_{\ell=0}^{\lfloor(p-r)/2\rfloor}
{p-2\ell-1\choose r-1},
\qquad 1\le r\le\min(k,p).
\]

The remaining dimension is (dim\ker B_p).

## Finite polynomial consequences

Let (R=\min(k,p)) and (L=L_p^-).  The orthogonal cycle projector is

\[
P_{\ker B_p}=\prod_{\lambda=2}^{R+1}
\left(I-\frac{L}{\lambda}\right).
\]

For matrices with (p\ge2), this simplifies to

\[
P_{\ker B_p}=I-\frac56L+\frac16L^2.
\]

If (P(x)=\prod_{\lambda=2}^{R+1}(1-x/\lambda)) and
(g(x)=(1-P(x))/x), then

\[
B_p^\dagger=g(L)B_p^*.
\]

The exact Sobolev resolvent ((I+\mu L)^{-1}) is likewise the unique
degree-(R) interpolation polynomial taking the values
(1/(1+\mu\lambda)) on
(lambda\in\{0,2,\ldots,R+1\}).

## Interpretation

The associated DSTM is contractible, so this is not a discovery of nonzero
homology or topological memory.  The contribution is an unusually simple
integer Hodge spectrum and the finite algorithms it enables.  The cycle
projector gives a valid finite form of boundary correction; ordinary repeated
(B^*B) feedback remains an iterative diffusion.
