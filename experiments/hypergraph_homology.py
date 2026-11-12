"""Compare exact embedded and closure homology on fixed finite hypergraphs."""

import argparse
import json
from itertools import combinations

from simplicial_tensors.hypergraph_homology import (
    associated_simplicial_complex, embedded_chain_complex, embedded_homology,
)


def examples():
    vertices3 = [(i,) for i in range(3)]
    triangle = vertices3 + [(0, 1), (1, 2), (0, 2)]
    facets = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 5), (0, 4, 5),
              (1, 2, 5), (1, 3, 4), (1, 4, 5), (2, 3, 4), (2, 3, 5)]
    return {
        'triangle': triangle,
        'filled_triangle': triangle + [(0, 1, 2)],
        'missing_faces': vertices3 + [(0, 1), (0, 1, 2)],
        'cancellation': [(i,) for i in range(4)] +
                        [(0, 1), (1, 2), (2, 3), (0, 3), (0, 1, 2), (0, 2, 3)],
        'field_dependent': sorted({face for facet in facets
                                   for size in range(1, 4) for face in combinations(facet, size)}),
    }


def main(argv=None):
    cases = examples()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modulus', type=int, help='Prime coefficient modulus; default Q')
    parser.add_argument('--case', choices=tuple(cases), help='Run just one fixed example')
    parser.add_argument('--representatives', action='store_true', help='Include lifted homology representatives')
    args = parser.parse_args(argv)
    output = []
    for name, edges in cases.items():
        if args.case is not None and name != args.case:
            continue
        summary = embedded_homology(edges, modulus=args.modulus)
        closure = embedded_homology(associated_simplicial_complex(edges), modulus=args.modulus)
        row = {'case': name, 'field': 'Q' if args.modulus is None else f'F_{args.modulus}',
               'chain_dimensions': summary.chain_dimensions,
               'boundary_ranks': summary.boundary_ranks,
               'embedded_betti': summary.homology_dimensions,
               'closure_betti': closure.homology_dimensions}
        if args.representatives:
            model = embedded_chain_complex(edges, modulus=args.modulus)
            row['representatives'] = []
            for degree in range(len(model.dimensions)):
                lifted = model.lift(degree, model.homology_representatives(degree))
                row['representatives'].append({
                    'degree': degree, 'hyperedges': model.ambient_bases[degree],
                    'columns': [[str(value) for value in lifted.col(column)]
                                for column in range(lifted.cols)],
                })
        output.append(row)
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    main()
