import pytest
from abstract_algebra.compound_structures.vector import Vector
from tests.fixtures.parameter_fixtures import parameter_vector, parameter_matrix


def test_vector_length(parameter_vector):
    assert len(parameter_vector) == len(
        parameter_vector.entries
    ), f"Length of Vector doesn't match the length of it's entries: {parameter_vector}"


def test_matrix_length(parameter_matrix):
    assert parameter_matrix.shape[0] == len(
        parameter_matrix.rows
    ), f"First Dimension of Matrix doesn't match the number of Rows: {parameter_matrix}"
    for row in parameter_matrix.rows:
        assert (
            parameter_matrix.shape[1] == row.dimension
        ), f"Second Dimension of Matrix doesn't match the length of a Row: {row}"


@pytest.mark.parametrize(
    "first,second,expected",
    [(Vector([1.0, 2.0]), Vector([2.0, 4.0]), Vector([3.0, 6.0]))],
)
def test_vector_addition(first: Vector, second: Vector, expected: Vector):
    result = first + second
    assert (
        expected == result
    ), f"Failure of Vector Addition. Added {first} to {second}. Expected: {expected}. Actual: {result}"
