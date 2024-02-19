from typing import TypeVar, Tuple
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.group import additive_inverse
from abstract_algebra.abstract_structures.ring import multiplicative_identity
from abstract_algebra.abstract_structures.field import (
    FieldProtocol,
    multiplicative_inverse,
)
from abstract_algebra.compound_structures.vector import identify_first_nonzero_entry
from abstract_algebra.compound_structures.matrix import Matrix, get_identity_matrix

F = TypeVar("F", bound=FieldProtocol)


def reduced_row_echelon_form(matrix: Matrix[F]) -> Matrix[F]:
    return reduced_row_echelon_form_track_inverse_and_determinants(matrix)[0]


def find_determinant(matrix: Matrix[F]) -> F:
    return reduced_row_echelon_form_track_inverse_and_determinants(matrix)[2]


def row_echelon_form(matrix: Matrix[F]) -> Matrix[F]:
    return _row_echelon_form_track_row_operations(matrix)[0]


def reduced_row_echelon_form_track_inverse_and_determinants(
    matrix: Matrix[F],
) -> Tuple[Matrix[F], Matrix[F], F]:
    """
    calculate the reduced row echelon form of a matrix
    also returns the "pseudo-inverse" and determinant

    consider the following equation:
    R = EA where
    R is the reduced row echelon form of A
    E is the matrix you need to left multiply A by to get R

    We consider E to be the "pseudo-inverse".
    Note that in the event A has an inverse,
    R will be the identity Matrix
    E will be the inverse of A

    det(A) is the determinant of the matrix A

    :param matrix: the input matrix (A) to reduce
    :return: R, E, det(A) (see above for the definition of these values)
    """
    upper_triangular_matrix, row_operations, swap_parity = (
        _row_echelon_form_track_row_operations(matrix)
    )

    diagonal_matrix, next_row_operations = (
        _diagonalize_row_echelon_form_track_row_operations(upper_triangular_matrix)
    )
    row_operations = next_row_operations @ row_operations

    rref_matrix, next_row_operations = _normalize_diagonal_matrix_track_row_operations(
        diagonal_matrix
    )
    row_operations = next_row_operations @ row_operations

    if upper_triangular_matrix.is_square():
        determinant = upper_triangular_matrix.diagonal_product()
        if swap_parity:
            determinant = additive_inverse(determinant)
    else:
        determinant = additive_identity(matrix[0][0])

    return rref_matrix, row_operations, determinant


def _row_echelon_form_track_row_operations(
    matrix: Matrix[F],
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


    :param matrix:
    :return: R, E, parity (see above for definition of these values)
    """

    zero = additive_identity(matrix[0][0])
    one = multiplicative_identity(matrix[0][0])

    row_operation_dimension = matrix.shape[0]
    row_operations = get_identity_matrix(row_operation_dimension, one)

    swap_count = 0
    pivot_row: int = 0
    for pivot_column in range(0, matrix.shape[0] - 1):
        column = matrix.transpose()[pivot_column]
        swap_row = identify_first_nonzero_entry(column, starting_index=pivot_row)

        if swap_row == -1:
            continue

        if swap_row != pivot_row:
            swap_count += 1
            row_swap_matrix = get_identity_matrix(matrix.shape[0], one)
            temp_row = row_swap_matrix[swap_row]
            row_swap_matrix[swap_row] = row_swap_matrix[pivot_row]
            row_swap_matrix[pivot_row] = temp_row
            matrix = row_swap_matrix @ matrix
            row_operations = row_swap_matrix @ row_operations

        pivot_value = matrix[pivot_row][pivot_column]
        pivot_matrix = Matrix(
            [
                [
                    (
                        one
                        if (j == i)
                        else (
                            zero
                            if (j != pivot_column or i < pivot_row)
                            else (zero - (matrix[i][j] / pivot_value))
                        )
                    )
                    for j in range(row_operation_dimension)
                ]
                for i in range(row_operation_dimension)
            ]
        )

        row_operations = pivot_matrix @ row_operations
        matrix = pivot_matrix @ matrix
        pivot_row += 1

    return matrix, row_operations, (swap_count % 2) != 0


def _diagonalize_row_echelon_form_track_row_operations(
    matrix: Matrix[F],
) -> Tuple[Matrix[F], Matrix[F]]:
    """
    Takes in a Row Echelon Form Matrix and returns a
    pseudo-diagonalized matrix along with the (invertible) transform
    it took to get there

    consider the following equation:
    D = ER
    where R is the input Row Echelon Form Matrix
    E is a linear matrix transformation
    D is a pseudo-diagonalized matrix

    In this context "Pseudo-diagonalized" refers to
    zero-ing out all the entries above the pivot entries
    of the input R. In the special case that R is upper triangular,
    This process will yield a Diagonal Matrix

    :param matrix: a matrix in Row Echelon Form
    :return: D, E (as defined above)
    """

    zero = additive_identity(matrix[0][0])
    one = multiplicative_identity(matrix[0][0])

    row_operation_dimension = matrix.shape[0]
    row_operations = get_identity_matrix(row_operation_dimension, one)
    for pivot_column in range(matrix.shape[1] - 1, -1, -1):
        column = matrix.transpose()[pivot_column]
        pivot_row = identify_first_nonzero_entry(
            column, starting_index=row_operation_dimension - 1, reverse=True
        )
        if pivot_row == -1:
            continue
        pivot_value = matrix[pivot_row][pivot_column]
        pivot_matrix = Matrix(
            [
                [
                    (
                        one
                        if (j == i)
                        else (
                            zero
                            if (j != pivot_row or i > pivot_row)
                            else (zero - (matrix[i][pivot_column] / pivot_value))
                        )
                    )
                    for j in range(row_operation_dimension)
                ]
                for i in range(row_operation_dimension)
            ]
        )
        row_operations = pivot_matrix @ row_operations
        matrix = pivot_matrix @ matrix

    return matrix, row_operations


def _normalize_diagonal_matrix_track_row_operations(
    matrix: Matrix[F],
) -> Tuple[Matrix[F], Matrix[F]]:
    """
    Takes in a "Pseudo-diagonal" matrix and returns a matrix
    with the pivots normalized to 1 along with the transform
    to get to there

    consider the following equation:
    R = ED where
    R is the reduced (normalized) matrix
    D is the pseudo-diagonal input matrix
    E is a linear matrix transformation

    In this context "Pseudo-diagonalized" refers to a matrix where we've
    zero-ed out all the entries above the pivot entries
    In the special case this is being done on an invertible matrix,
    The pseudo-diagonalized matrix is diagonal

    :param matrix: the pseudo-diagonalized matrix
    :return: R, E (as defined above)
    """

    zero = additive_identity(matrix[0][0])
    one = multiplicative_identity(matrix[0][0])

    pivot_indexes = [
        identify_first_nonzero_entry(matrix[i]) for i in range(matrix.shape[0])
    ]

    scaling_matrix = Matrix(
        [
            [
                (
                    zero
                    if i != j
                    else (
                        one
                        if pivot_indexes[i] == -1
                        else multiplicative_inverse(matrix[i][pivot_indexes[i]])
                    )
                )
                for j in range(matrix.shape[0])
            ]
            for i in range(matrix.shape[0])
        ]
    )

    return scaling_matrix @ matrix, scaling_matrix
