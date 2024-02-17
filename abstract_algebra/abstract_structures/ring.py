from typing import Protocol, Self, TypeVar, runtime_checkable, cast
from abstract_algebra.abstract_structures.group import GroupProtocol


@runtime_checkable
class RingProtocol(GroupProtocol, Protocol):
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


@runtime_checkable
class RingExplicitIdentity(RingProtocol, Protocol):
    def get_multiplicative_identity(self) -> Self:
        raise NotImplementedError(f"Not Implemented")


R = TypeVar("R", bound=RingProtocol)


def multiplicative_identity(r: R) -> R:
    if isinstance(r, int):
        return cast(R, 1)
    elif isinstance(r, float):
        return cast(R, 1.0)
    elif isinstance(r, RingExplicitIdentity):
        return cast(R, r.get_multiplicative_identity())
    else:
        try:
            return cast(R, getattr(r, "__truediv__")(r))
        except Exception as e:
            raise TypeError(
                f"Unable to find multiplicative identity for Ring of type '{type(r)}': {r}."
                f"Ring needs to be an int, float, or equipped with .get_multiplicative_identity or /"
                f"to find multiplicative identity"
            )
