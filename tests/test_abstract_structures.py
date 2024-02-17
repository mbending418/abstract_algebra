from abstract_algebra.abstract_structures.monoid import MonoidProtocol
from abstract_algebra.abstract_structures.group import GroupProtocol
from abstract_algebra.abstract_structures.ring import RingProtocol
from abstract_algebra.abstract_structures.euclidean_ring import EuclideanRingProtocol
from abstract_algebra.abstract_structures.field import FieldProtocol
from tests.fixtures.parameter_fixtures import (
    parameter_monoid,
    parameter_group,
    parameter_ring,
    parameter_euclidean_ring,
    parameter_field,
)


def test_monoid(parameter_monoid: MonoidProtocol):
    assert isinstance(
        parameter_monoid, MonoidProtocol
    ), f"test parameter should implement MonoidProtocol: {parameter_monoid}"


def test_group(parameter_group: GroupProtocol):
    assert isinstance(
        parameter_group, GroupProtocol
    ), f"test parameter should implement GroupProtocol: {parameter_group}"


def test_ring(parameter_ring: RingProtocol):
    assert isinstance(
        parameter_ring, RingProtocol
    ), f"test parameter should implement RingProtocol: {parameter_ring}"


def test_euclidean_ring(parameter_euclidean_ring: EuclideanRingProtocol):
    assert isinstance(
        parameter_euclidean_ring, EuclideanRingProtocol
    ), f"test parameter should implement EuclidianRingProtocol: {parameter_euclidean_ring}"


def test_field(parameter_field: FieldProtocol):
    assert isinstance(
        parameter_field, FieldProtocol
    ), f"test parameter should implement FieldProtocol: {parameter_field}"
