# Symbolic tensor tests

[tests/test_symbolic_tensor_ops.py](../tests/test_symbolic_tensor_ops.py) tests [simplicial_tensors.symbolic_tensor_ops](../src/simplicial_tensors/symbolic_tensor_ops.py) with NumPy object arrays and SymPy expressions. Pytest skips the module if either dependency cannot be imported.

## Coverage

The fourteen tests cover entry names, face and degeneracy shapes, boundary construction, agreement between included horn faces and a filler, and the type and nonnegativity of tensor_filler_difference_rank.

Two tests check comparison control flow. A (2, 2) tensor has no inner horns, so the default comparison returns True. A monkeypatch changes one earlier inner-horn filler while preserving a later one and checks that the method reports the discrepancy.

The remaining tests check the five families of simplicial identities and boundary squared equal to zero. They use bounded random shapes and seed Python's random module from tensor_ops.___SEED___.

The function tensor_filler_difference_rank counts distinct simplified difference expressions. Its test does not establish their linear independence.

## Running the tests

From the repository root after installing the package and pytest:

~~~sh
python -m pytest -q tests/test_symbolic_tensor_ops.py
~~~

The current module passes all fourteen tests with the SymPy backend. Output goes to the pytest console, with no experiment data files.

~~~sh
python -m pytest -q
~~~

The second command runs the repository suite.
