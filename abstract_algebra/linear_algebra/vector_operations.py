from typing import TypeVar
import functools
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.field import FieldProtocol
from abstract_algebra.compound_structures.vector import Vector


F = TypeVar("F", bound=FieldProtocol)


def dot_product(v: Vector[F], w: Vector[F]) -> F:
    return functools.reduce(
        lambda a, b: a + b,
        [xi * yi for (xi, yi) in zip(v.entries, w.entries)],
        additive_identity(v.entries[0]),
    )


def identify_first_nonzero_entry(
    vector: Vector[F], starting_index: int = 0, reverse: bool = False
) -> int:
    """
    identify the first nonzero entry starting at "starting_index"

    :param vector: the vector
    :param starting_index: the first index to check
    :param reverse: set to True to look backwards through the vector instead
    :return: returns the index of the first nonzero entry. returns -1 as a Sentinel Value if none is found
    """

    zero = additive_identity(vector[0])
    if not reverse:
        range_bounds = [starting_index, len(vector)]
    else:
        range_bounds = [starting_index, -1, -1]
    for index in range(*range_bounds):
        if vector[index] != zero:
            return index
    return -1
