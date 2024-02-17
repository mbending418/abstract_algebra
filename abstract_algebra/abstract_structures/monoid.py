from typing import Protocol, Self, TypeVar, runtime_checkable, cast


@runtime_checkable
class MonoidProtocol(Protocol):
    def __eq__(self, other) -> bool:
        raise NotImplementedError(f"'==' not implemented for {type(self)}")

    def __add__(self: Self, other: Self) -> Self:
        raise NotImplementedError(f"'+' not implemented for {type(self)}")


@runtime_checkable
class MonoidExplicitIdentity(MonoidProtocol, Protocol):
    def get_additive_identity(self) -> Self:
        raise NotImplementedError(f"Not Implemented")


M = TypeVar("M", bound=MonoidProtocol)


def additive_identity(m: M) -> M:
    if isinstance(m, int):
        return cast(M, 0)
    elif isinstance(m, float):
        return cast(M, 0.0)
    elif isinstance(m, MonoidExplicitIdentity):
        return cast(M, m.get_additive_identity())
    else:
        try:
            return cast(M, getattr(m, "__sub__")(m))
        except Exception as e:
            raise TypeError(
                f"Unable to find additive identity for Monoid of type '{type(m)}': {m}."
                f"Monoid needs to be an int, float, or equipped with .get_additive_identity or -"
                f"to find additive identity"
            )
