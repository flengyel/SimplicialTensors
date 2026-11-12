"""Reproducible example outputs and exact characteristic-dependent jumps."""

import json
from pathlib import Path
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[1]


def experiment():
    return runpy.run_path(str(ROOT / "experiments/kernel_sequences.py"))


@pytest.mark.parametrize("modulus", [None, 2, 3, 5])
def test_parameter_family_homology_depends_on_t_minus_one(modulus):
    results = experiment()["examples"](modulus=modulus, case="jump")["examples"]["jump"]
    for parameter, record in results.items():
        value = int(parameter) - 1
        nonzero = value != 0 if modulus is None else value % modulus != 0
        assert record["betti_numbers"] == ((0, 1, 0) if nonzero else (0, 0, 0))
        assert record["kernel_dimensions"] == ((2, 0, 0) if nonzero else (2, 1, 0))


@pytest.mark.parametrize("modulus", [None, 2, 3])
def test_marked_kernel_realization_lifting_and_obstructions(modulus):
    records = experiment()["examples"](modulus=modulus, max_degree=2)["examples"]
    assert not records["marked"]["same_kernel"]
    assert records["realize"]["same_kernel"]
    assert records["lifting"]["invertible"]
    assert records["lifting"]["ranks"] == [3, 3, 1]
    assert records["obstruction"]["excluded_plane"]["degree"] == 1
    assert records["obstruction"]["excluded_plane"]["required"] == 3
    assert records["obstruction"]["excluded_plane"]["available"] == 2
    assert records["obstruction"]["positive_tail_boundary"]["degree"] == 0
    assert records["obstruction"]["positive_tail_boundary"]["required"] == 4
    assert records["obstruction"]["positive_tail_boundary"]["available"] == 3


def test_sparse_sphere_examples_retain_all_universal_source_degrees():
    results = experiment()["examples"](case="spheres", max_degree=4)["examples"]["spheres"]
    for n in range(1, 5):
        record = results[str(n)]
        assert record["stored_entries"] == 1
        assert record["betti_numbers"] == (0,) * n + (1,)
        assert len(record["kernel_dimensions"]) == n + 1


def test_cli_serializes_exact_matrices_and_validates_options(capsys):
    module = experiment()
    assert module["main"](["--case", "marked", "--matrices"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["examples"]["marked"]["T"]["kernel_inclusions"][0]["shape"] == [3, 2]
    for arguments in (["--modulus", "4"], ["--max-degree", "0"]):
        with pytest.raises(SystemExit) as error:
            module["main"](arguments)
        assert error.value.code == 2


def test_import_has_no_output_or_files(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    experiment()
    assert not capsys.readouterr().out
    assert not list(tmp_path.iterdir())
