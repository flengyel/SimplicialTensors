"""Count vertex orders by the mod-two rank of the interleaved adjacency boundary."""

from collections import Counter
from math import factorial

if __package__:
    from .interleaved_graph_paths import examples, hamiltonian_orders
else:
    from interleaved_graph_paths import examples, hamiltonian_orders


def automorphisms(neighbors):
    """Enumerate graph automorphisms as tuples of images of the vertex labels."""
    n = len(neighbors)
    full = (1 << n) - 1
    images = []

    def extend(unused):
        vertex = len(images)
        if not unused:
            yield tuple(images)
            return
        candidates = unused
        for previous, image in enumerate(images):
            if neighbors[vertex] & (1 << previous):
                candidates &= neighbors[image]
            else:
                candidates &= ~neighbors[image]
        while candidates:
            bit = candidates & -candidates
            candidates ^= bit
            images.append(bit.bit_length() - 1)
            yield from extend(unused ^ bit)
            images.pop()

    yield from extend(full)


def relabel(mask, image_bits):
    """Apply a vertex permutation to a set of vertices or a binary vector."""
    result = 0
    while mask:
        bit = mask & -mask
        result |= image_bits[bit.bit_length() - 1]
        mask ^= bit
    return result


def subset_representatives(neighbors):
    """For each subset, find an orbit representative and a permutation to it."""
    n = len(neighbors)
    actions = []
    for images in automorphisms(neighbors):
        forward = tuple(1 << image for image in images)
        inverse = [0] * n
        for vertex, image in enumerate(images):
            inverse[image] = 1 << vertex
        actions.append((forward, tuple(inverse)))

    transports = [None] * (1 << n)
    for representative in range(1 << n):
        if transports[representative] is not None:
            continue
        for forward, inverse in actions:
            subset = relabel(representative, forward)
            if transports[subset] is None:
                transports[subset] = (representative, inverse)
    return transports


def interleaved_distribution(adjacency):
    """Map boundary rank to its frequency among all orders of a simple graph."""
    n = len(adjacency)
    neighbors = [
        sum(1 << j for j, entry in enumerate(row) if entry)
        for row in adjacency
    ]
    transports = subset_representatives(neighbors)
    # State: remaining vertices, residual row space on them, prefix rank.
    states = Counter({((1 << n) - 1, (0,), 0): 1})
    for _ in range(n):
        next_states = Counter()
        for (remaining, space, rank), multiplicity in states.items():
            choices = remaining
            while choices:
                bit = choices & -choices
                choices ^= bit
                vertex = bit.bit_length() - 1
                rest = remaining ^ bit
                kernel = [row for row in space if row & bit == 0]
                next_rank = rank + (len(kernel) < len(space))
                new_row = neighbors[vertex] & rest
                if new_row not in kernel:
                    kernel += [row ^ new_row for row in kernel]

                representative, inverse = transports[rest]
                next_space = tuple(sorted(relabel(row, inverse) for row in kernel))
                next_states[representative, next_space, next_rank] += multiplicity
        states = next_states

    # The interleaved boundary has twice the rank of the forward adjacency.
    return {2 * rank: count for (_, _, rank), count in states.items()}


def main():
    results = {}
    for name, graph, _ in examples():
        counts = interleaved_distribution(graph)
        n = len(graph)
        assert sum(counts.values()) == factorial(n)
        assert counts.get(2 * n - 2, 0) == hamiltonian_orders(graph)
        results[name] = counts

    print("Boundary rank | " + " | ".join(results))
    for rank in sorted(set().union(*results.values())):
        print(f"{rank} | " + " | ".join(f"{d.get(rank, 0):,}" for d in results.values()))
    print("Total | " + " | ".join(f"{sum(d.values()):,}" for d in results.values()))


if __name__ == "__main__":
    main()
