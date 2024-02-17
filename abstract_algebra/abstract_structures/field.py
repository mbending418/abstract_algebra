from typing import Protocol, Self, runtime_checkable, TypeVar
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.ring import (
    RingProtocol,
    multiplicative_identity,
)


@runtime_checkable
class FieldProtocol(RingProtocol, Protocol):
    def __eq__(self, other) -> bool:
        raise NotImplementedError(f"'==' not implemented for {type(self)}")

    def __add__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'+' not implemented for {type(self)}")

    def __sub__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'-' not implemented for {type(self)}")

    def __rsub__(self: Self, other: Self) -> Self:
        return other - self

    def __mul__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'*' not implemented for {type(self)}")

    def __truediv__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'/' not implemented for {type(self)}")

    def __rtruediv__(self: Self, other: Self) -> Self:
        return other / self


F = TypeVar("F", bound=FieldProtocol)


def multiplicative_inverse(f: F) -> F:
    if f == additive_identity(f):
        raise ZeroDivisionError(
            f"Cannot find the multiplicative inverse of the additive identity of the field: {f}"
        )
    return multiplicative_identity(f) / f
