"""Shared shape and storage validation for symbolic tensor interfaces."""

from operator import index

import numpy as np


def symbolic_tensor_arguments(shape, tensor):
    """Return a validated shape and an optional copied object array."""
    try:
        dimensions = tuple(shape)
    except TypeError as exc:
        raise TypeError("The shape must be a sequence of integers.") from exc
    if not dimensions:
        raise ValueError("A symbolic tensor must have at least one axis.")
    checked = []
    for dimension in dimensions:
        if isinstance(dimension, (bool, np.bool_)):
            raise TypeError("Shape entries must be integers.")
        try:
            dimension = index(dimension)
        except TypeError as exc:
            raise TypeError("Shape entries must be integers.") from exc
        if dimension < 0:
            raise ValueError("Shape entries must be nonnegative.")
        checked.append(dimension)
    shape = tuple(checked)
    array = None if tensor is None else np.array(tensor, dtype=object, copy=True)
    if array is not None and array.shape != shape:
        raise ValueError("The supplied tensor data must match the stated shape.")
    return shape, array
