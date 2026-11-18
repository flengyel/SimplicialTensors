"""Compute the two-axis tensor horn for the tree a -> e, (e,b) -> r.

Both axes are indexed by edges, with no additional indices. Matrix units
are indexed by ordered pairs. The omitted face contracts the inner edge e.
"""

import json
from itertools import product


def two_vertex_tensor_horn() -> dict:
    """Return the ranks of the two restrictions and the missing matrix units."""
    tree = set(product(("a", "e", "b", "r"), repeat=2))
    upper_face = set(product(("a", "e"), repeat=2))
    lower_face = set(product(("e", "b", "r"), repeat=2))
    common_subface = set(product(("e",), repeat=2))
    included = upper_face | lower_face

    # A horn prescribes one value on each face, agreeing on the common subface.
    matching_rank = len(upper_face) + len(lower_face) - len(common_subface)
    horn_kernel = tree - included

    corolla_edges = ("a", "b", "r")
    contracted_face = set(product(corolla_edges, repeat=2))
    # Each elementary face of the corolla retains just one edge.
    corolla_kernel = contracted_face - {(edge, edge) for edge in corolla_edges}
    kernel_image = horn_kernel & contracted_face
    missing_units = sorted(corolla_kernel - kernel_image)

    return {
        "horn_restriction": {
            "domain_rank": len(tree),
            "codomain_rank": matching_rank,
            "image_rank": len(included),
            "kernel_rank": len(horn_kernel),
            "cokernel_rank": matching_rank - len(included),
        },
        "omitted_face_restriction_on_kernels": {
            "domain_rank": len(horn_kernel),
            "codomain_rank": len(corolla_kernel),
            "image_rank": len(kernel_image),
            "kernel_rank": len(horn_kernel - contracted_face),
            "cokernel_rank": len(missing_units),
            "cokernel_matrix_units": missing_units,
        },
    }


if __name__ == "__main__":
    print(json.dumps(two_vertex_tensor_horn(), indent=2))
