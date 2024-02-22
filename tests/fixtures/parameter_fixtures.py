import pytest
from typing import List, cast
from abstract_algebra.abstract_structures.monoid import MonoidProtocol
from abstract_algebra.abstract_structures.group import GroupProtocol
from abstract_algebra.abstract_structures.ring import RingProtocol
from abstract_algebra.abstract_structures.euclidean_ring import EuclideanRingProtocol
from abstract_algebra.abstract_structures.field import FieldProtocol
from abstract_algebra.compound_structures.vector import Vector
from abstract_algebra.compound_structures.matrix import Matrix
from abstract_algebra.compound_structures.fraction import Fraction
from abstract_algebra.concrete_structures.complex import ComplexNumber, GaussianInteger

integer_values: List[int] = [0, 4, 10, 23]
float_values: List[float] = [0.5, 2.4, 4.0, 120.4]
fraction_int_values: List[Fraction[int]] = [
    Fraction(-1),
    Fraction(1),
    Fraction(1, 2),
    Fraction(-2, 3),
    Fraction(12, 77),
    Fraction(123, 55555),
]
fraction_complex_values: List[Fraction[GaussianInteger]] = [
    Fraction(GaussianInteger(1, 0)),
    Fraction(GaussianInteger(0, 1)),
    Fraction(GaussianInteger(1, 2), GaussianInteger(3, 5)),
    Fraction(GaussianInteger(-1, 7), GaussianInteger(5, 11)),
]
complex_values: List[ComplexNumber] = [
    ComplexNumber(0.5, 0.5),
    ComplexNumber(20.33, 20.1),
    ComplexNumber(20.33, 0.0),
]
gaussian_integer_values: List[GaussianInteger] = [
    GaussianInteger(1, 0),
    GaussianInteger(0, 1),
    GaussianInteger(2, 2),
    GaussianInteger(4, 3),
    GaussianInteger(3, 4),
]
vector_values: List[Vector] = [
    Vector((1.0, 0.0, 0.0)),
    Vector((1.0, 0.0, 0.0, 0.0)),
    Vector((0.0, 1.0, 0.0)),
    Vector((0.0, 0.0, 2.0, 0.0)),
    Vector((1.0, 2.0, 3.0, 4.0)),
    Vector(
        (ComplexNumber(1.0, 1.0), ComplexNumber(1.0, 0.0), ComplexNumber(0.0, -1.0))
    ),
    Vector((Fraction[int](1, 2), Fraction[int](1, -1), Fraction[int](0, 1))),
    Vector(
        (
            Fraction[GaussianInteger](GaussianInteger(1)),
            Fraction[GaussianInteger](GaussianInteger(1, 1), GaussianInteger(-3, 5)),
        )
    ),
]
matrix_values: List[Matrix] = [
    Matrix.new_matrix([[1.0, 0.0], [0.0, 1.0]]),
    Matrix.new_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
    Matrix.new_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    Matrix.new_matrix([[-1.5, -2.0, 4.0]]),
    Matrix.new_matrix([[1.0, 2.0, 3.0]]),
    Matrix.new_matrix([[-1.0, -2.5, 2.5], [10.4, -1.52, 1.234]]),
    Matrix.new_matrix(
        [
            [ComplexNumber(1.0, 1.0), ComplexNumber(-1.0, 0.0)],
            [ComplexNumber(0.0, 1.0), ComplexNumber(0.0, 0.0)],
        ]
    ),
    Matrix.new_matrix(
        [
            [Fraction[int](1, 2), Fraction[int](-3, 1)],
            [Fraction[int](5, 7), Fraction[int](5)],
        ]
    ),
    Matrix.new_matrix(
        [
            [
                Fraction[GaussianInteger](GaussianInteger(1)),
                Fraction[GaussianInteger](GaussianInteger(0, 1), GaussianInteger(2, 2)),
            ],
            [
                Fraction[GaussianInteger](
                    GaussianInteger(11, -13), GaussianInteger(3, 5)
                ),
                Fraction[GaussianInteger](GaussianInteger(-2, 3)),
            ],
        ]
    ),
]

field_values: List[FieldProtocol] = cast(
    List[FieldProtocol],
    float_values + complex_values + fraction_int_values + fraction_complex_values,
)
euclidian_ring_values: List[EuclideanRingProtocol] = cast(
    List[EuclideanRingProtocol],
    float_values + integer_values + gaussian_integer_values,
)
ring_values: List[RingProtocol] = cast(
    List[RingProtocol], field_values + gaussian_integer_values + integer_values
)
group_values: List[GroupProtocol] = cast(
    List[GroupProtocol], ring_values + vector_values + matrix_values
)
monoid_values: List[GroupProtocol] = cast(List[GroupProtocol], group_values)


@pytest.fixture(params=integer_values)
def parameter_integer(request) -> int:
    return request.param


@pytest.fixture(params=float_values)
def parameter_float(request) -> float:
    return request.param


@pytest.fixture(params=complex_values)
def parameter_complex(request) -> ComplexNumber:
    return request.param


@pytest.fixture(params=gaussian_integer_values)
def parameter_gaussian(request) -> GaussianInteger:
    return request.param


@pytest.fixture(params=fraction_int_values)
def parameter_fraction_int(request) -> Fraction[int]:
    return request.param


@pytest.fixture(params=fraction_complex_values)
def parameter_fraction_complex(request) -> Fraction[GaussianInteger]:
    return request.param


@pytest.fixture(params=vector_values)
def parameter_vector(request) -> Vector[FieldProtocol]:
    return request.param


@pytest.fixture(params=matrix_values)
def parameter_matrix(request) -> Matrix[FieldProtocol]:
    return request.param


@pytest.fixture(params=monoid_values)
def parameter_monoid(request) -> MonoidProtocol:
    return request.param


@pytest.fixture(params=group_values)
def parameter_group(request) -> GroupProtocol:
    return request.param


@pytest.fixture(params=ring_values)
def parameter_ring(request) -> RingProtocol:
    return request.param


@pytest.fixture(params=euclidian_ring_values)
def parameter_euclidean_ring(request) -> EuclideanRingProtocol:
    return request.param


@pytest.fixture(params=field_values)
def parameter_field(request) -> FieldProtocol:
    return request.param
