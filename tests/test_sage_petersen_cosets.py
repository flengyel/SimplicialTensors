"""Bounded Sage/GAP regressions for the Petersen transversal action."""

from collections import Counter
from itertools import permutations
from pathlib import Path
import runpy

import pytest

sage = pytest.importorskip("sage.all")
SCRIPT = Path(__file__).resolve().parents[1] / "experiments/notebooks/petersen_coset_enumeration.sage"
experiment = runpy.run_path(str(SCRIPT))


def test_gap_representative_conversion_uses_the_inverse_image_list():
    g = sage.libgap.PermList([2, 3, 1, 4])
    assert experiment["gap_perm_to_list"](g, 4) == [2, 0, 1, 3]


def test_right_cosets_cover_all_labelings_of_a_small_path():
    graph = sage.graphs.PathGraph(4)
    T = sage.matrix(sage.GF(2), graph.adjacency_matrix())
    expected = {}
    for perm in permutations(range(4)):
        relabeled = T.matrix_from_rows_and_columns(perm, perm)
        expected[tuple(relabeled.list())] = int(experiment["boundary_mod2"](relabeled).rank())
    order, cosets, distinct, counts, weighted = experiment["right_coset_rank_distribution"](graph)
    assert order == 2
    assert cosets == distinct == len(expected) == 12
    assert counts == dict(Counter(expected.values()))
    assert weighted == {r: 2 * c for r, c in counts.items()}


def test_shared_boolean_boundary_matches_native_sage_deletions():
    T = sage.matrix(sage.GF(2), [[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    expected = sage.zero_matrix(sage.GF(2), 2, 2)
    for deleted in range(3):
        survivors = [i for i in range(3) if i != deleted]
        expected += T.matrix_from_rows_and_columns(survivors, survivors)
    assert experiment["boundary_mod2"](T) == expected
