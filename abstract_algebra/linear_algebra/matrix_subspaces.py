from typing import TypeVar, List, Generic, Tuple
from dataclasses import dataclass
import functools
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.group import additive_inverse
from abstract_algebra.abstract_structures.ring import multiplicative_identity
from abstract_algebra.abstract_structures.field import FieldProtocol
from abstract_algebra.compound_structures.vector import Vector
from abstract_algebra.compound_structures.matrix import Matrix
from abstract_algebra.linear_algebra import vector_operations
from abstract_algebra.linear_algebra.gauss_jordan import GaussJordan

F = TypeVar("F", bound=FieldProtocol)


@dataclass(init=True, frozen=True)
class MatrixSubspaces(Generic[F]):
    matrix: Matrix[F]

    @functools.cached_property
    def reduced_row_echelon_form(self) -> Matrix[F]:
        return GaussJordan(self.matrix).reduced_row_echelon_form

    @functools.cached_property
    def column_space(self) -> List[Vector[F]]:
        row_count = self.matrix.shape[0]
        reduced_matrix = self.reduced_row_echelon_form
        basic_indexes = [
            index
            for i in range(row_count)
            if (
                index := vector_operations.identify_first_nonzero_entry(
                    reduced_matrix[i]
                )
            )
            != -1
        ]
        return [
            column
            for index, column in enumerate(self.matrix.transpose().rows)
            if index in basic_indexes
        ]

    @functools.cached_property
    def null_space(self) -> List[Vector[F]]:
        zero = additive_identity(self.matrix[0][0])
        one = multiplicative_identity(self.matrix[0][0])
        row_count = self.matrix.shape[0]
        column_count = self.matrix.shape[1]
        reduced_matrix = self.reduced_row_echelon_form
        basic_indexes = [
            index
            for i in range(row_count)
            if (
                index := vector_operations.identify_first_nonzero_entry(
                    reduced_matrix[i]
                )
            )
            != -1
        ]
        free_indexes = [j for j in range(column_count) if j not in basic_indexes]
        null_space_vectors: List[Vector[F]] = []
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
            null_space_vectors.append(Vector.new_vector(column_vector))
        return null_space_vectors
