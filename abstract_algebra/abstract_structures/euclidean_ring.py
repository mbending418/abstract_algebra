from typing import Protocol, Self, TypeVar, runtime_checkable
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.group import additive_inverse
from abstract_algebra.abstract_structures.ring import RingProtocol


@runtime_checkable
class EuclideanRingProtocol(RingProtocol, Protocol):
    def __eq__(self, other) -> bool:
        raise NotImplementedError(f"'==' not implemented for {type(self)}")

    def __gt__(self: Self, other: Self) -> bool:
        raise NotImplementedError(f"'>' not implemented for {type(self)}")

    def __lt__(self: Self, other: Self) -> bool:
        raise NotImplementedError(f"'<' not implemented for {type(self)}")

    def __add__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'+' not implemented for {type(self)}")

    def __sub__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'-' not implemented for {type(self)}")

    def __rsub__(self: Self, other: Self) -> Self:
        return other - self

    def __mul__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'*' not implemented for {type(self)}")

    def __floordiv__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'//' not implemented for {type(self)}")

    def __rfloordiv__(self: Self, other: Self) -> Self:
        return other // self

    def __mod__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'%' not implemented for {type(self)}")


E = TypeVar("E", bound=EuclideanRingProtocol)


def generalized_gcd(numerator: E, denominator: E) -> E:
    zero = additive_identity(denominator)
    if numerator < zero:
        numerator = additive_inverse(numerator)
    if denominator < zero:
        denominator = additive_inverse(denominator)
    if denominator == zero:
        return numerator
    elif numerator < denominator:
        return generalized_gcd(denominator, numerator)
    else:
        return generalized_gcd(numerator % denominator, denominator)
