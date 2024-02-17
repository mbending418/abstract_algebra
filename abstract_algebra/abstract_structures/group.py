from typing import Protocol, Self, runtime_checkable, TypeVar
from abstract_algebra.abstract_structures.monoid import (
    MonoidProtocol,
    additive_identity,
)


@runtime_checkable
class GroupProtocol(MonoidProtocol, Protocol):
    def __eq__(self, other) -> bool:
        raise NotImplementedError(f"'==' not implemented for {type(self)}")

    def __add__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'+' not implemented for {type(self)}")

    def __sub__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'-' not implemented for {type(self)}")

    def __rsub__(self: Self, other: Self) -> Self:
        return other - self


G = TypeVar("G", bound=GroupProtocol)


def additive_inverse(g: G) -> G:
    return additive_identity(g) - g
