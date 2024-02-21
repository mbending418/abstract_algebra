from typing import TypeVar, List

from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.group import additive_inverse
from abstract_algebra.abstract_structures.ring import multiplicative_identity
from abstract_algebra.abstract_structures.field import FieldProtocol
from abstract_algebra.compound_structures.vector import (
    Vector,
    identify_first_nonzero_entry,
)
from abstract_algebra.compound_structures.matrix import Matrix
from abstract_algebra.linear_algebra.gauss_jordan import GaussJordan

F = TypeVar("F", bound=FieldProtocol)


def column_space(matrix: Matrix[F]) -> List[Vector[F]]:
    row_count = matrix.shape[0]
    reduced_matrix = GaussJordan(matrix).reduced_row_echelon_form
    basic_indexes = [
        index
        for i in range(row_count)
        if (index := identify_first_nonzero_entry(reduced_matrix[i])) != -1
    ]
    return [
        column
        for index, column in enumerate(matrix.transpose().rows)
        if index in basic_indexes
    ]


def null_space(matrix: Matrix[F]) -> List[Vector[F]]:
    zero = additive_identity(matrix[0][0])
    one = multiplicative_identity(matrix[0][0])
    row_count = matrix.shape[0]
    column_count = matrix.shape[1]
    reduced_matrix = GaussJordan(matrix).reduced_row_echelon_form
    basic_indexes = [
        index
        for i in range(row_count)
        if (index := identify_first_nonzero_entry(reduced_matrix[i])) != -1
    ]
    free_indexes = [j for j in range(column_count) if j not in basic_indexes]
    null_space_vectors = []
    for free_variable_index in free_indexes:
        column_vector = []
        column_entry_pointer = 0
        for column_index in range(column_count):
            if column_index in basic_indexes:
                column_vector.append(
                    additive_inverse(
                        reduced_matrix[column_entry_pointer][free_variable_index]
                    )
                )
                column_entry_pointer += 1
            elif column_index == free_variable_index:
                column_vector.append(one)
            else:
                column_vector.append(zero)
        null_space_vectors.append(Vector(column_vector))
    return null_space_vectors


def in_span(vectors: List[Vector[F]], v: Vector[F]) -> bool:
    column_matrix = Matrix(vectors).transpose()
    column_basis = column_space(column_matrix)

    augmented_column_matrix = Matrix(vectors + [v]).transpose()
    augmented_column_basis = column_space(augmented_column_matrix)

    return len(column_basis) == len(augmented_column_basis)
