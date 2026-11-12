"""Enumerate a finite set of shapes with explicit degree and size bounds."""

import argparse
import operator


def _integer_at_least(value, minimum, name):
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def generate_shapes_by_dimension(
    max_total_shapes=None,
    *,
    min_degree=2,
    max_degree=4,
    max_axes=4,
    max_offset_sum=2,
):
    """Yield shapes in degree, axis-count, offset-sum, lexicographic order.

    A shape of degree ``n`` has at least ``n`` axes and minimum entry
    ``n + 1``. Its entries are ``n + 1 + d_j``, with nonnegative offsets
    of minimum zero and sum at most ``max_offset_sum``. All bounds are
    finite. ``None`` yields the whole bounded set; zero yields no shapes.
    """
    min_degree = _integer_at_least(min_degree, 1, "min_degree")
    max_degree = _integer_at_least(max_degree, min_degree, "max_degree")
    max_axes = _integer_at_least(max_axes, max_degree, "max_axes")
    max_offset_sum = _integer_at_least(max_offset_sum, 0, "max_offset_sum")
    if max_total_shapes is not None:
        max_total_shapes = _integer_at_least(max_total_shapes, 0, "max_total_shapes")
        if max_total_shapes == 0:
            return

    count = 0
    for degree in range(min_degree, max_degree + 1):
        for axes in range(degree, max_axes + 1):
            for offset_sum in range(max_offset_sum + 1):
                for offsets in _compositions_non_negative_lex(offset_sum, axes):
                    if min(offsets) != 0:
                        continue
                    yield tuple(degree + 1 + offset for offset in offsets)
                    count += 1
                    if max_total_shapes is not None and count >= max_total_shapes:
                        return


def _compositions_non_negative_lex(total, parts):
    """Yield compositions of ``total`` into ``parts`` nonnegative integers."""
    if parts < 1:
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions_non_negative_lex(total - first, parts - 1):
            yield (first,) + rest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=35, help="maximum number of shapes (default: 35)")
    parser.add_argument("--min-degree", type=int, default=2)
    parser.add_argument("--max-degree", type=int, default=4)
    parser.add_argument("--max-axes", type=int, default=4)
    parser.add_argument("--max-offset-sum", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        for shape in generate_shapes_by_dimension(
            args.limit,
            min_degree=args.min_degree,
            max_degree=args.max_degree,
            max_axes=args.max_axes,
            max_offset_sum=args.max_offset_sum,
        ):
            print(f"degree={min(shape) - 1}, axes={len(shape)}, shape={shape}")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
