from math import isclose
from typing import cast, SupportsFloat
from abstract_algebra.abstract_structures.monoid import (
    additive_identity,
    MonoidProtocol,
)
from abstract_algebra.abstract_structures.group import (
    additive_inverse,
    GroupProtocol,
)
from abstract_algebra.abstract_structures.ring import (
    multiplicative_identity,
    RingProtocol,
)
from abstract_algebra.abstract_structures.field import (
    multiplicative_inverse,
    FieldProtocol,
)
from tests.fixtures.parameter_fixtures import (
    parameter_monoid,
    parameter_group,
    parameter_ring,
    parameter_field,
)


def test_equals_self(parameter_monoid: MonoidProtocol):
    assert (
        parameter_monoid == parameter_monoid
    ), f"Monoid element doesn't equal itself: {parameter_monoid}"


def test_additive_identity_works(parameter_monoid: MonoidProtocol):
    identity = additive_identity(parameter_monoid)
    assert (
        parameter_monoid + identity == parameter_monoid
    ), f"Additive Identity doesn't act as Identity: {parameter_monoid}"


def test_additive_inverse_works(parameter_group: GroupProtocol):
    identity = additive_identity(parameter_group)
    inverse = additive_inverse(parameter_group)
    assert (
        parameter_group + inverse == identity
    ), f"Additive Inverse doesn't act as Inverse: {parameter_group}"


def test_multiplicative_identity_works(parameter_ring: RingProtocol):
    identity = multiplicative_identity(parameter_ring)
    assert (
        parameter_ring * identity == parameter_ring
    ), f"Multiplicative Identity doesn't act as Identity: {parameter_ring}"


def test_multiplicative_inverse_works(parameter_field: FieldProtocol):
    identity = multiplicative_identity(parameter_field)
    inverse = multiplicative_inverse(parameter_field)
    value_x_inverse = parameter_field * inverse
    if isinstance(value_x_inverse, SupportsFloat):
        res = isclose(value_x_inverse, cast(SupportsFloat, identity))
    else:
        res = value_x_inverse == identity
    assert res, f"Multiplicative Inverse doesn't act as Inverse: {parameter_field}"
