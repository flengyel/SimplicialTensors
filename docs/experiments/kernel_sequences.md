# Kernel-sequence examples

[kernel_sequences.py](../../experiments/kernel_sequences.py) computes the
universal kernels of tensor-generated modules and exercises the two lifting
constructions in the [API](../api/kernel_sequences.md). Run from the repository
root with the package installed:

```sh
python experiments/kernel_sequences.py
python experiments/kernel_sequences.py --modulus 3 --case jump
python experiments/kernel_sequences.py --case marked --matrices
python experiments/kernel_sequences.py --modulus 2 --case spheres --max-degree 4
python experiments/kernel_sequences.py --case realize
python experiments/kernel_sequences.py --case lifting --matrices
python experiments/kernel_sequences.py --case obstruction
```

The program writes JSON to standard output and creates no files. The default
field is the rationals; `--modulus p` selects a prime field. All inputs are fixed.
`--matrices` includes exact realization matrices, source and row labels, kernel
inclusions, and kernel differentials. Rational coefficients are serialized as
strings. `--parameter t` adds an integer value to the matrix-family calculation.

## Matrix kernels and a homology jump

The family is
\[
T(t)=\begin{pmatrix}0&1&0\\0&1&1\\1&1&t\end{pmatrix}.
\]
Its normalized degree-one realization determinant is $t-1$. If $t\ne1$ in the
selected field, the kernel dimensions are $(2,0,0)$, quotient dimensions are
$(1,3,1)$, boundary ranks are $(0,1,1)$, and Betti numbers are $(0,1,0)$.
At $t=1$, the kernel dimensions are $(2,1,0)$ and the quotient is acyclic.
Thus $T(4)$ has first Betti number one over the rationals and zero in
characteristic three. The default values are $0,1,2,4$.

The `marked` case compares $T(2)$ with $T(2)+J$, where $J$ is the all-ones
matrix. Their kernel planes are $\ker(0,1,2)$ and $\ker(1,2,3)$, respectively.
The kernels differ, while the quotient chain complexes are isomorphic.

## Higher homology, realization, and lifting

The `spheres` case uses the degree-$q$ coordinate tensor $E_{(0,1,\ldots,q)}$
with $q+1$ axes and empty tails. Every face is zero, so its generated normalized
complex has one generator in degree $q$ and no other generators. It stores one
nonzero tensor entry. `--max-degree` defaults to three; the universal source
still grows exponentially with this degree.

The `realize` case constructs a tensor from the kernel of $T(2)$ and checks
equality of all kernel subspaces after reconstruction. The output includes its
sparse coordinates. The `lifting` case chooses an isomorphism between the
quotients for $T(2)$ and $T(2)+J$, lifts it to an automorphism of the universal
source, and checks the quotient-map identity in every degree.

## Capacity obstructions

The `obstruction` case reports two failed embeddings with explicit dimensions.
The proposed kernel $L_0=\ker(1,1,1)$, $L_1=L_2=0$ gives a quotient with three
degree-one cycles. The square-matrix ambient complex has only two such cycles,
so no tensor in this ambient module has exactly that kernel.

For the positive-tail example, four identity complexes in degrees one and zero
have four independent boundaries in degree zero. A disk model of
$N(X((1,1);K))$ has dimensions $(4,5,2)$, cycle dimensions $(4,2,0)$, and boundary
dimensions $(3,2,0)$. The cycle inequalities hold, but the proposed four
boundaries exceed the three available. This demonstrates the need to check
boundary capacities as well as cycle capacities.

The [finite dendroidal program](dendroidal_representable_chains.md) provides a
cycle-lifting obstruction: its relative top class has connecting class
$[u+v-w]$ in the homology of the tree boundary.
