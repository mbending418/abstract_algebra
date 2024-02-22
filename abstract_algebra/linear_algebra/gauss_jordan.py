from typing import TypeVar, Tuple, Generic, List
from dataclasses import dataclass
import functools
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.group import additive_inverse
from abstract_algebra.abstract_structures.ring import multiplicative_identity
from abstract_algebra.abstract_structures.field import (
    FieldProtocol,
    multiplicative_inverse,
)
from abstract_algebra.compound_structures.vector import Vector
from abstract_algebra.compound_structures.matrix import Matrix
from abstract_algebra.linear_algebra import vector_operations
from abstract_algebra.linear_algebra import matrix_operations

F = TypeVar("F", bound=FieldProtocol)


@dataclass(init=True)
class GaussJordan(Generic[F]):
    base_matrix: Matrix[F]

    @property
    def reduced_row_echelon_form(self) -> Matrix[F]:
        return self._reduced_row_echelon_form_and_transformation_matrix[0]

    @property
    def pseudo_inverse(self) -> Matrix[F]:
        return self._reduced_row_echelon_form_and_transformation_matrix[1]

    @functools.cached_property
    def determinant(self) -> F:
        if not matrix_operations.is_square(self.row_echelon_form):
            return self._zero
        determinant = matrix_operations.diagonal_product(self.row_echelon_form)
        if self._row_echelon_transformation_parity:
            return additive_inverse(determinant)
        else:
            return determinant

    @property
    def pseudo_diagonal(self) -> Matrix[F]:
        return self._pseudo_diagonal_form_and_transformation_matrix[0]

    @property
    def row_echelon_form(self) -> Matrix[F]:
        return self._row_echelon_form_and_transformation_matrix_and_parity[0]

    @functools.cached_property
    def _zero(self) -> F:
        return additive_identity(self.base_matrix[0][0])

    @functools.cached_property
    def _one(self) -> F:
        return multiplicative_identity(self.base_matrix[0][0])

    @property
    def _row_echelon_transformation_matrix(self) -> Matrix[F]:
        return self._row_echelon_form_and_transformation_matrix_and_parity[1]

    @property
    def _row_echelon_transformation_parity(self) -> bool:
        return self._row_echelon_form_and_transformation_matrix_and_parity[2]

    @functools.cached_property
    def _row_echelon_form_and_transformation_matrix_and_parity(
        self,
    ) -> Tuple[Matrix[F], Matrix[F], bool]:
        """
        reduce the matrix to a row echelon form (not reduced)
        also returns the "inverse" of the reduction, and it's row swap parity

        consider the following equation:
        R = EA
        where R is a row echelon form of A
        E is the matrix to left multiply A by to get R

        the "parity" or "row swap parity" is how many times modulo 2
        that the rows were swapped during the transformation

        :return: R, E, parity (see above for definition of these values)
        """

        result_matrix = self.base_matrix

        row_operation_dimension = result_matrix.shape[0]
        identity_matrix = matrix_operations.identity_matrix(
            dimensions=row_operation_dimension, example_field_element=self._one
        )
        row_operations = identity_matrix

        swap_count = 0
        pivot_row: int = 0
        for pivot_column in range(0, result_matrix.shape[0] - 1):
            column = result_matrix.transpose()[pivot_column]
            swap_row = vector_operations.identify_first_nonzero_entry(
                column, starting_index=pivot_row
            )

            if swap_row == -1:
                continue

            if swap_row != pivot_row:
                swap_count += 1
                row_swap_list: List[Vector[F]] = list(identity_matrix.rows)
                temp_row: Vector[F] = row_swap_list[swap_row]
                row_swap_list[swap_row] = row_swap_list[pivot_row]
                row_swap_list[pivot_row] = temp_row
                row_swap_matrix: Matrix[F] = Matrix.new_matrix(row_swap_list)
                result_matrix = row_swap_matrix @ result_matrix
                row_operations = row_swap_matrix @ row_operations

            pivot_value = result_matrix[pivot_row][pivot_column]
            pivot_matrix: Matrix[F] = Matrix.new_matrix(
                [
                    [
                        (
                            self._one
                            if (j == i)
                            else (
                                self._zero
                                if (j != pivot_column or i < pivot_row)
                                else (self._zero - (result_matrix[i][j] / pivot_value))
                            )
                        )
                        for j in range(row_operation_dimension)
                    ]
                    for i in range(row_operation_dimension)
                ]
            )

            row_operations = pivot_matrix @ row_operations
            result_matrix = pivot_matrix @ result_matrix
            pivot_row += 1

        return result_matrix, row_operations, (swap_count % 2) != 0

    @functools.cached_property
    def _pseudo_diagonal_form_and_transformation_matrix(
        self,
    ) -> Tuple[Matrix[F], Matrix[F]]:
        """
        reduce the matrix to pseudo-diagonalized form
        also returns the "inverse" of the (invertible) transform

        consider the following equation:
        D = EA
        where E is a linear matrix transformation
        D is a pseudo-diagonalized matrix

        In this context "Pseudo-diagonalized" refers to
        putting the matrix in row echelon form and then
        zero-ing out all the entries above the pivot entries.
        In the special case that A is invertible,
        This process will yield a Diagonal Matrix

        :return: D, E (as defined above)
        """

        result_matrix = self.row_echelon_form
        row_operations = self._row_echelon_transformation_matrix

        row_operation_dimension = result_matrix.shape[0]
        for pivot_column in range(result_matrix.shape[1] - 1, -1, -1):
            column = result_matrix.transpose()[pivot_column]
            pivot_row = vector_operations.identify_first_nonzero_entry(
                column, starting_index=row_operation_dimension - 1, reverse=True
            )
            if pivot_row == -1:
                continue
            pivot_value = result_matrix[pivot_row][pivot_column]
            pivot_matrix: Matrix[F] = Matrix.new_matrix(
                [
                    [
                        (
                            self._one
                            if (j == i)
                            else (
                                self._zero
                                if (j != pivot_row or i > pivot_row)
                                else (
                                    self._zero
                                    - (result_matrix[i][pivot_column] / pivot_value)
                                )
                            )
                        )
                        for j in range(row_operation_dimension)
                    ]
                    for i in range(row_operation_dimension)
                ]
            )
            row_operations = pivot_matrix @ row_operations
            result_matrix = pivot_matrix @ result_matrix

        return result_matrix, row_operations

    @functools.cached_property
    def _reduced_row_echelon_form_and_transformation_matrix(
        self,
    ) -> Tuple[Matrix[F], Matrix[F]]:
        """
        calculate the reduced row echelon form of a matrix
        also returns the "pseudo-inverse"

        consider the following equation:
        R = EA where
        R is the reduced row echelon form of A
        E is the matrix you need to left multiply A by to get R

        We consider E to be the "pseudo-inverse".
        Note that in the event A has an inverse,
        R will be the identity Matrix
        E will be the inverse of A

        :return: R, E (see above for the definition of these values)
        """
        starting_matrix = self.pseudo_diagonal
        row_operations = self._pseudo_diagonal_form_and_transformation_matrix[1]

        pivot_indexes = [
            vector_operations.identify_first_nonzero_entry(starting_matrix[i])
            for i in range(starting_matrix.shape[0])
        ]

        scaling_matrix: Matrix[F] = Matrix.new_matrix(
            [
                [
                    (
                        self._zero
                        if i != j
                        else (
                            self._one
                            if pivot_indexes[i] == -1
                            else multiplicative_inverse(
                                starting_matrix[i][pivot_indexes[i]]
                            )
                        )
                    )
                    for j in range(starting_matrix.shape[0])
                ]
                for i in range(starting_matrix.shape[0])
            ]
        )

        return scaling_matrix @ starting_matrix, scaling_matrix @ row_operations
