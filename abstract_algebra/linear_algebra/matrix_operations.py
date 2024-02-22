from typing import Tuple, TypeVar
import functools
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.ring import multiplicative_identity
from abstract_algebra.abstract_structures.field import FieldProtocol
from abstract_algebra.compound_structures.vector import Vector
from abstract_algebra.compound_structures.matrix import Matrix


F = TypeVar("F", bound=FieldProtocol)


def zero_matrix(shape: Tuple[int, int], example_field_element: F) -> Matrix[F]:
    zero = additive_identity(example_field_element)

    return Matrix.new_matrix([[zero for i in range(shape[0])] for j in range(shape[1])])


def identity_matrix(dimensions: int, example_field_element: F) -> Matrix[F]:
    zero = additive_identity(example_field_element)
    one = multiplicative_identity(example_field_element)
    return Matrix.new_matrix(
        [
            [one if i == j else zero for j in range(dimensions)]
            for i in range(dimensions)
        ]
    )


def matrix_to_vector(matrix: Matrix[F]) -> Vector[F]:
    if matrix.shape[0] == 1:
        return matrix.rows[0]
    elif matrix.shape[1] == 1:
        return matrix_to_vector(matrix.transpose())
    else:
        raise TypeError(
            f"Cannot convert Matrix[{matrix.field}] of size {matrix.shape} to a Vector"
        )


def vector_to_matrix(vector: Vector[F]) -> Matrix[F]:
    return Matrix.new_matrix([vector])


def is_square(matrix: Matrix[F]) -> bool:
    return matrix.shape[0] == matrix.shape[1]


def trace(matrix: Matrix[F]) -> F:
    if not is_square(matrix):
        raise TypeError(
            f"Cannot calculate Trace of non-square matrix: shape={matrix.shape}"
        )
    return functools.reduce(
        lambda a, b: a + b,
        [matrix[i][i] for i in range(matrix.shape[0])],
        additive_identity(matrix[0][0]),
    )


def diagonal_product(matrix: Matrix[F]) -> F:
    if not is_square(matrix):
        raise TypeError(
            f"Cannot calculate diagonal product of non-square matrix: shape={matrix.shape}"
        )
    return functools.reduce(
        lambda a, b: a * b,
        [matrix[i][i] for i in range(matrix.shape[0])],
        multiplicative_identity(matrix[0][0]),
    )
