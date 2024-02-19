from typing import TypeVar, List

from abstract_algebra.abstract_structures.field import FieldProtocol
from abstract_algebra.compound_structures.vector import Vector
from abstract_algebra.compound_structures.matrix import Matrix
from abstract_algebra.linear_algebra.gauss_jordan import reduced_row_echelon_form

F = TypeVar("F", bound=FieldProtocol)

def identify_pivot_columns

def null_space(matrix: Matrix[F]) -> List[Vector[F]]:
    reduced_matrix = reduced_row_echelon_form(matrix)
