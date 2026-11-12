# Symbolic mapping-cone prototype

[experiments/sagemath_normalized_tensor_homology.py](../experiments/sagemath_normalized_tensor_homology.py) defines a standalone `SymbolicTensor` class and mapping-cone routines using NumPy and SymPy. It imports no SageMath modules. It has no main function or executable example, so direct execution prints nothing.

## Available functions

The local `SymbolicTensor` initializes independent entry symbols by default and zero entries with `init_type="zeros"`. It supports addition, subtraction, faces, and degeneracies.

The `build_tensor_basis` function collects iterated faces and degeneracies through degree `max_dim + 1`. Its keys preserve entry order. The `build_differential_matrices` function forms alternating face matrices on these symbols. The `compute_homology_rank` function subtracts the outgoing and incoming matrix ranks. The `compute_relative_homology_rank` function constructs the formal horn generators, inclusion, and mapping cone.

## Interpretation

Each distinct encountered tensor becomes an independent basis symbol. Linear dependencies in the ambient tensor module are not imposed. A zero tensor can therefore appear as a nonzero formal basis element. The resulting ranks cannot be identified with homology of the tensor-generated submodule without a comparison that accounts for these relations.

The differential matrices are unnormalized alternating-face matrices on those symbols. The degree-zero helper also infers the chain-group dimension from the outgoing differential, which is absent in degree zero. It needs an explicit degree-zero dimension to compute that homology.

The functions are a retained prototype. The source comments calling it “definitive” do not establish the missing comparison.

## Inspecting the prototype

From the repository root with its dependencies installed:

~~~python
import runpy

module = runpy.run_path("experiments/sagemath_normalized_tensor_homology.py")
T = module["SymbolicTensor"]((2, 2))
basis = module["build_tensor_basis"]([T], 1)
print({degree: len(tensors) for degree, tensors in basis.items()})
~~~

The current manuscript's exact quotient construction is in [generated_section.tex](../tex/generated_section.tex). A runnable matrix example and its required software revision are documented in [tex/anc/README.txt](../tex/anc/README.txt).
