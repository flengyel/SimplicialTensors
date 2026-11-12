"""Exact finite representable chains with Basic--Nikolaus face signs."""

from pathlib import Path
import runpy

import pytest
import sympy as sp


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = runpy.run_path(str(ROOT / "experiments/dendroidal_representable_chains.py"))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_representables_have_leaf_homology_and_relative_cell_classes(modulus):
    models = EXPERIMENT["representable_models"](modulus=modulus)
    for name, model in models.items():
        summary = model.homology()
        if name.startswith("relative_"):
            assert summary.homology_dimensions == (0,) * (len(model.dimensions) - 1) + (1,)
        else:
            assert summary.homology_dimensions == (2,) + (0,) * (len(model.dimensions) - 1)
        for degree in range(1, len(model.dimensions)):
            assert not any(model.field.matmul(model.differential(degree - 1),
                                              model.differential(degree)))
        for degree, betti in enumerate(summary.homology_dimensions):
            representatives = model.homology_representatives(degree)
            boundaries = model.boundaries(degree)
            assert representatives.cols == betti
            assert not any(model.field.matmul(model.differential(degree), representatives))
            assert model.field.rank(boundaries.row_join(representatives)) == boundaries.cols + betti


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_one_vertex_boundaries_are_inputs_minus_output(modulus):
    models = EXPERIMENT["representable_models"](modulus=modulus)
    for name, operations in {
        "binary_corolla": {"C2": (("a", "b"), "r")},
        "unary_binary_tree": {"u": (("a",), "e"), "v": (("e", "b"), "r"),
                              "w": (("a", "b"), "r")},
    }.items():
        model = models[name]
        colours, cells = model.ambient_bases[:2]
        expected = sp.Matrix([[operations[cell][0].count(colour) - int(operations[cell][1] == colour)
                               for cell in cells] for colour in colours])
        assert model.differential(1) == model.field.matrix(expected)


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_relative_lift_is_a_representative_modulo_the_boundary(modulus):
    models = EXPERIMENT["representable_models"](modulus=modulus)
    absolute = models["unary_binary_tree"]
    relative = models["relative_unary_binary_tree"]
    representative = relative.homology_representatives(2)
    lifted = relative.lift(2, representative)
    assert relative.ambient_bases[2] == ("T",)
    assert relative.field.rank(lifted) == 1
    assert relative.field.rank(relative.field.matmul(absolute.differential(2), lifted)) == 1
    assert not any(relative.field.matmul(relative.differential(2), representative))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_boundaries_and_betti_match_native_sage(modulus):
    sage = pytest.importorskip("sage.all")
    field = sage.QQ if modulus is None else sage.GF(modulus)
    models = EXPERIMENT["representable_models"](modulus=modulus)
    for model in models.values():
        ranks = []
        for boundary in model.differentials:
            native = sage.matrix(field, boundary.rows, boundary.cols,
                                 [field(int(value)) for value in boundary])
            ranks.append(int(native.rank()))
        expected_betti = tuple(dimension - ranks[degree] -
                               (ranks[degree + 1] if degree + 1 < len(ranks) else 0)
                               for degree, dimension in enumerate(model.dimensions))
        assert tuple(ranks) == model.homology().boundary_ranks
        assert expected_betti == model.homology().homology_dimensions


def test_cli_validates_the_prime_field(capsys):
    with pytest.raises(SystemExit) as error:
        EXPERIMENT["main"](["--modulus", "4"])
    assert error.value.code == 2
    assert "prime" in capsys.readouterr().err
