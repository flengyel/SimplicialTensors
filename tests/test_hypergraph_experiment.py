"""Finite example output and import safety for the hypergraph entry point."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('modulus,expected', [(None, (1, 0, 0)), (2, (1, 1, 1)), (3, (1, 0, 0))])
def test_hypergraph_cli_field_dependent_fixture(modulus, expected):
    command = [sys.executable, str(ROOT / 'experiments/hypergraph_homology.py'),
               '--case', 'field_dependent', '--representatives']
    if modulus is not None:
        command += ['--modulus', str(modulus)]
    result = subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True, timeout=45)
    row, = json.loads(result.stdout)
    assert row['chain_dimensions'] == [6, 15, 10]
    assert tuple(row['embedded_betti']) == expected
    assert tuple(row['closure_betti']) == expected
    assert [len(value['columns']) for value in row['representatives']] == list(expected)


def test_hypergraph_import_does_not_execute(capsys):
    module = runpy.run_path(str(ROOT / 'experiments/hypergraph_homology.py'))
    assert callable(module['main'])
    assert capsys.readouterr().out == ''
