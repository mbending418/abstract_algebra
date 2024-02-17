from typing import (
    TypeVar,
    Generic,
    List,
    Type,
    Any,
    Optional,
    Callable,
    Iterable,
    Iterator,
    overload,
    cast,
)


from functools import reduce
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.field import (
    FieldProtocol,
    multiplicative_inverse,
)

F = TypeVar("F", bound=FieldProtocol)
T = TypeVar("T", bound=FieldProtocol)


class Vector(Generic[F], Iterable):
    entries: List[F]
    field: Type  # The Type of F
    dimension: int

    @overload
    def __init__(self, entries: Iterable[F], field_factory: None = None):
        pass

    @overload
    def __init__(self, entries: Iterable[Any], field_factory: Callable[[Any], F]):
        pass

    def __init__(
        self, entries: Iterable[Any], field_factory: Optional[Callable[[Any], F]] = None
    ):
        if field_factory is None:
            self.entries: List[F] = list(entries)
        else:
            self.entries: List[F] = [field_factory(entry) for entry in entries]
        self.field = type(self.entries[0])
        self.dimension = len(self.entries)
        for entry in self.entries:
            if not isinstance(entry, self.field):
                raise TypeError(
                    f"All entries of the vector need to be of the same type: Mismatched types: "
                    f"{type(entry)} | {self.field}"
                )

    def __repr__(self) -> str:
        return (
            f"abstract_algebra.modules.Vector[{self.field}]"
            f"[{','.join([entry.__repr__() for entry in self.entries])}]"
        )

    def __str__(self) -> str:
        return f"[{", ".join([entry.__str__() for entry in self.entries])}]"

    def __getitem__(self, index: int) -> F:
        return self.entries[index]

    def __setitem__(self, index: int, value: F):
        self.entries[cast(int, index)] = value

    def __iter__(self) -> Iterator[F]:
        for entry in self.entries:
            yield entry

    def __len__(self) -> int:
        return self.dimension

    def convert_to(
        self, field_factory: Optional[Callable[[T], F]] = None
    ) -> "Vector[F]":
        return Vector(entries=self.entries, field_factory=field_factory)

    def _validate_scalar_operation(self, scalar: Any, operator: str) -> bool:
        if not isinstance(scalar, self.field):
            raise TypeError(
                f"unsupported operand type(s) for {operator}: "
                f"'Vector[{self.field}]' and '{type(scalar)}'"
            )
        return True

    def _validate_elementwise_operation(self, other: Any, operator: str) -> bool:
        if not isinstance(other, Vector):
            raise TypeError(
                f"unsupported operand type(s) for {operator}: 'Vector[{self.field}]' and '{type(other)}'"
            )
        elif self.field != other.field:
            raise TypeError(
                f"unsupported operand type(s) for {operator}:"
                f"'Vector[{self.field}]' and 'Vector[{other.field}]'"
            )
        elif len(self) != len(other):
            raise TypeError(
                f"unsupported operand type(s) for {operator}: "
                f"'Dim(Vector[{self.field}])={len(self)}' and 'Dim(Vector[{other.field}])={len(other)}'"
            )
        return True

    def __eq__(self, other) -> bool:
        self._validate_elementwise_operation(other=other, operator="==")
        return all([x == y for x, y in zip(self.entries, other.entries)])

    def __add__(self, other: "Vector[F]") -> "Vector[F]":
        self._validate_elementwise_operation(other=other, operator="+")
        return Vector([x + y for x, y in zip(self.entries, other.entries)])

    def __sub__(self, other: "Vector[F]") -> "Vector[F]":
        self._validate_elementwise_operation(other=other, operator="-")
        return Vector([x - y for x, y in zip(self.entries, other.entries)])

    def __rsub__(self, other: "Vector[F]") -> "Vector[F]":
        return other - self

    def __mul__(self, scalar: F) -> "Vector[F]":
        self._validate_scalar_operation(scalar=scalar, operator="*")
        return Vector([x * scalar for x in self.entries])

    def __rmul__(self, scalar: F) -> "Vector[F]":
        self._validate_scalar_operation(scalar=scalar, operator="/")
        return self * scalar

    def __truediv__(self, scalar: F) -> "Vector[F]":
        return self * multiplicative_inverse(scalar)

    # dot product between vectors
    def __pow__(self, other: "Vector[F]") -> F:
        return reduce(
            lambda a, b: a + b,
            [xi * yi for (xi, yi) in zip(self.entries, other.entries)],
            additive_identity(self.entries[0]),
        )
