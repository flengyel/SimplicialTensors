"""Compute the two-axis horn of a -> e, (e,b) -> h, h -> r, omitting contraction of h.

Both axes are indexed by the edges. Restrictions must agree on each common
face. A single-edge face with colour e identifies only the coordinate (e, e).
"""

import json
from itertools import combinations, product


def composite(lower, upper):
    """Substitute the upper operation into the matching input of the lower one."""
    output, inputs = lower
    upper_output, upper_inputs = upper
    other_inputs = set(inputs) - {upper_output}
    return output, tuple(sorted(other_inputs.union(upper_inputs)))


def proper_faces(lower, upper):
    """Return the edges and the three one-vertex faces of a two-vertex tree."""
    corollas = {lower, upper, composite(lower, upper)}
    edges = {edge for output, inputs in corollas for edge in (output, *inputs)}
    return edges, corollas


def three_vertex_tensor_horn():
    """Compute the ranks of horn restriction and list the disconnected tuples."""
    top = ("e", ("a",))
    middle = ("h", ("b", "e"))
    bottom = ("r", ("h",))
    faces = {
        "prune_top": proper_faces(bottom, middle),
        "contract_e": proper_faces(bottom, composite(middle, top)),
        "prune_root": proper_faces(middle, top),
    }
    shared = {}
    for left, right in combinations(faces, 2):
        left_edges, left_corollas = faces[left]
        right_edges, right_corollas = faces[right]
        # Each common edge gives a single-entry restriction. Each common
        # corolla gives all ordered pairs of its colours. Profiles determine
        # the operations here because the input colours are distinct and this
        # tree has no nullary vertices.
        pairs = {(e, e) for e in left_edges & right_edges}
        for output, inputs in left_corollas & right_corollas:
            pairs.update(product((output, *inputs), repeat=2))
        shared[left, right] = pairs

    edges = sorted({edge for output, inputs in (top, middle, bottom)
                    for edge in (output, *inputs)})
    matching_rank = 0
    image_rank = 0
    disconnected = []
    for unit in product(edges, repeat=2):
        adjacent = {name: set() for name, (colours, _) in faces.items()
                    if all(edge in colours for edge in unit)}
        for (left, right), pairs in shared.items():
            if unit in pairs:
                adjacent[left].add(right)
                adjacent[right].add(left)
        remaining = set(adjacent)
        image_rank += bool(remaining)
        components = 0
        while remaining:
            components += 1
            stack = [remaining.pop()]
            while stack:
                neighbours = adjacent[stack.pop()] & remaining
                remaining.difference_update(neighbours)
                stack.extend(neighbours)
        matching_rank += components
        if components > 1:
            disconnected.append(unit)
    domain_rank = len(edges) ** 2
    return {
        "domain_rank": domain_rank,
        "matching_rank": matching_rank,
        "image_rank": image_rank,
        "kernel_rank": domain_rank - image_rank,
        "cokernel_rank": matching_rank - image_rank,
        "tuples_with_disconnected_faces": disconnected,
    }


if __name__ == "__main__":
    print(json.dumps(three_vertex_tensor_horn(), indent=2))
