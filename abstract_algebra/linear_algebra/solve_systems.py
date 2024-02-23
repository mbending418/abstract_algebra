from typing import TypeVar, Tuple, List, Optional
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.group import additive_inverse
from abstract_algebra.abstract_structures.field import FieldProtocol, multiplicative_inverse
from abstract_algebra.compound_structures.vector import Vector
from abstract_algebra.compound_structures.matrix import Matrix
from abstract_algebra.linear_algebra.matrix_subspaces import MatrixSubspaces

F = TypeVar("F", bound=FieldProtocol)


def in_null_space(matrix: Matrix[F], vector: Vector[F]) -> bool:
    zero = additive_identity(vector)
    return matrix @ vector == zero


def in_span(vectors: List[Vector[F]], v: Vector[F]) -> bool:
    column_matrix: Matrix[F] = Matrix.new_matrix(vectors).transpose()
    column_basis = MatrixSubspaces(column_matrix).column_space

    augmented_column_matrix: Matrix[F] = Matrix.new_matrix(vectors + [v]).transpose()
    augmented_column_basis = MatrixSubspaces(augmented_column_matrix).column_space

    return len(column_basis) == len(augmented_column_basis)


def solve_linear_system(matrix: Matrix[F], b: Vector[F]) -> Optional[Vector[F]]:
    augmented_matrix: Matrix[F] = Matrix.new_matrix(list(matrix.transpose().rows) + [b]).transpose()
    null_basis = MatrixSubspaces(augmented_matrix).null_space
    for vec in null_basis:
        if (k := vec[-1]) != 0:
            x = Vector.new_vector(vec.entries[:-1])
            return additive_inverse(multiplicative_inverse(k)) * x
    return None


def completely_solve_linear_system(
    matrix: Matrix[F], b: Vector[F]
) -> Tuple[Optional[Vector[F]], List[Vector[F]]]:
    return solve_linear_system(matrix, b), MatrixSubspaces(matrix).null_space
