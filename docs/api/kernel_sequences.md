# Kernel sequences and exact chain maps

[kernel_sequences.py](../../src/simplicial_tensors/kernel_sequences.py) retains
the universal presentation of a tensor-generated simplicial module. The module
[chain_maps.py](../../src/simplicial_tensors/chain_maps.py) supplies exact chain
embeddings, isomorphisms, and lifts over the rationals or a prime field.

## The universal presentation

Fix a field $K$ and a tensor $T$ in degree $n$. Let
$P=K[\Delta^n]$ be the free simplicial module on the standard simplex,
$C=N(P)$ its normalized complex, and $g_T:C\to N(X)$ the normalized realization
map. The normalized kernel sequence is $L_q(T)=\ker g_{T,q}$ for
$0\leq q\leq n$. The source basis in degree $q$ is indexed by increasing
$(q+1)$-tuples in $\{0,\ldots,n\}$, and
$\dim C_q=\binom{n+1}{q+1}$. Its differential is the alternating vertex-deletion
boundary. These finitely many kernels determine the full simplicial kernel.

`simplex_chain_complex(n, *, modulus=None)` returns this based source complex.
`kernel_sequence(tensor, *, modulus=None)` returns an immutable `KernelSequence`.
Dense input has degree `min(shape)-1`; a `SparseTensor` retains its explicit
degree and tail lengths. The coefficient rules are those of
[ExactField](chain_complex.md), including exact Sage `ZZ` and `QQ`
coefficients and rejection of real-field coefficients. The simplex degree and
the tail lengths supplied to `realize_kernel` also accept exact Sage integers.

The result retains:

| Attribute | Meaning |
|---|---|
| `tensor` | The sparse input, with coefficients interpreted in the selected field |
| `source` | The full complex $C$ in degrees $0,\ldots,n$ |
| `g_matrices`, `tensor_bases` | All realization columns and their finite tensor-coordinate row labels |
| `kernel` | $L(T)$, with its induced differential; `kernel.lifts` gives its inclusions into $C$ |
| `quotient` | A based model of $C/L(T)$; its lifts choose representatives in $C$ |
| `image` | A based tensor-coordinate model of the generated normalized complex |
| `inclusion`, `projection` | Chain maps $L(T)\to C$ and $C\to C/L(T)$ |
| `realization`, `comparison` | Chain maps $C\to\operatorname{im}g_T$ and $C/L(T)\to\operatorname{im}g_T$ |

The comparison is an isomorphism, and its composition with the projection is
the realization. The source, kernel and quotient retain every degree from zero
to $n$, including degrees in which the realization is zero. The image model may
omit trailing zero terms. A basis for the image alone does not retain the
relations between the original source columns.

```python
import numpy as np
from simplicial_tensors.kernel_sequences import kernel_sequence

T = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 2]], dtype=object)
presentation = kernel_sequence(T)
assert presentation.source.dimensions == (3, 3, 1)
assert presentation.kernel.dimensions == (2, 0, 0)
assert presentation.quotient.homology().homology_dimensions == (0, 1, 0)
```

Here $L_0(T)=\ker(0,1,2)$. For $T+J$, where $J$ is the all-ones matrix,
the degree-zero kernel is $\ker(1,2,3)$. Both quotients have the displayed
homology. These are kernels of the realization maps; `ChainComplex.cycles(q)`
instead computes the kernel of a differential.

## Embedding complexes and realizing kernels

For finite chain complexes $Q$ and $D$ over the same field $K$, write
$Z_q(Q)=\ker d_q$ and $B_q(Q)=\operatorname{im}d_{q+1}$. These are ordinary cycles, including
$Z_0(Q)=Q_0$. An injective chain map $Q\to D$ exists over a field exactly when
both $\dim Z_q(Q)\leq\dim Z_q(D)$ and
$\dim B_q(Q)\leq\dim B_q(D)$ hold in every degree.

`chain_embedding(Q, D)` constructs such a `ChainMap`. If an inequality fails,
`CapacityError` reports its degree, kind, required dimension, and available
dimension. The construction chooses injections between boundary and cycle
subspaces, then lifts the remaining basis vectors through the differentials.
`chain_isomorphism(Q, D)` constructs an isomorphism when the two finite based
complexes have the same dimensions and differential ranks.

`realize_kernel(inclusion, shifts)` accepts an injective `ChainMap` into
`simplex_chain_complex(n)`, with its ordered vertex-subset bases. It returns a
`SparseTensor` in degree $n$, with
the supplied tail lengths and exactly the prescribed normalized kernel.
The coefficient field is taken from the inclusion. The construction uses
selected coordinate cycles and their boundary sections, and reconstructs the
tensor from its normalized face images.

```python
from simplicial_tensors.kernel_sequences import realize_kernel

realized = realize_kernel(presentation.inclusion, (0, 0))
again = kernel_sequence(realized, modulus=presentation.source.modulus)
assert again.kernel.dimensions == presentation.kernel.dimensions
for left, right in zip(presentation.kernel.lifts, again.kernel.lifts):
    assert presentation.source.field.rank(left.row_join(right)) == left.cols == right.cols
```

The returned tensor depends on basis choices. It realizes the requested kernel,
which need not determine the original embedded tensor. Use the same modulus when
computing a kernel sequence of a tensor realized over a prime field.
Positive tail lengths on
every axis can contribute degree-zero homology to the ambient module; the
boundary-capacity condition is still required in that case.

## Lifting a specified quotient isomorphism

A `ChainMap(source, target, matrices)` checks matrix dimensions, coefficient
fields, and the chain-map identity. Its matrices are immutable.
`f.compose(g)` means $f\circ g$, and `f.inverse()` requires an isomorphism.
Composition uses the based chain coordinates; ambient lift labels do not alter
these maps.

`lift_quotient_isomorphism(pi_left, pi_right, phi)` takes two surjective chain
maps from the same finite projective complex and a specified isomorphism between
their quotients. It returns an automorphism `a` of the source such that
`pi_right.compose(a)` and `phi.compose(pi_left)` have identical matrices in every
degree.
For finite complexes of vector spaces, projectivity means that all positive
homology groups vanish; the standard simplex complex has this property.

```python
from simplicial_tensors.chain_maps import chain_isomorphism, lift_quotient_isomorphism

other = kernel_sequence(T + 1)
phi = chain_isomorphism(presentation.quotient, other.quotient)
a = lift_quotient_isomorphism(presentation.projection, other.projection, phi)
assert a.is_isomorphism()
```

The optional `forward_lift` and `backward_lift` arguments supply initial chain
lifts of `phi` and its inverse. Such a lift can be singular; the Fitting
construction corrects it to an invertible lift inducing the specified quotient
map. A quotient homology class has an absolute cycle lift precisely when its
connecting class in the kernel subcomplex's homology vanishes.

## Size of the calculation

The universal source has $2^{n+1}-1$ basis elements in total. Exact nullspaces,
quotient bases, and chain-map lifts can therefore become expensive as $n$
increases. Sparse tensor input avoids allocating its dense ambient array, while
degeneracies can expand its finite support. Kernel realization selects the
coordinate cycles it needs rather than constructing every ambient coordinate.

See the [example guide](../experiments/kernel_sequences.md) for parameter jumps,
higher homology, reconstruction, quotient lifts, and failed capacity conditions.
