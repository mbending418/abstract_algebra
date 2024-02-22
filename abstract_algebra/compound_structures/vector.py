from typing import (
    TypeVar,
    Generic,
    Tuple,
    Optional,
    Type,
    Any,
    Callable,
    Iterable,
    Iterator,
    cast,
)
from dataclasses import dataclass
import functools
from abstract_algebra.abstract_structures.field import (
    FieldProtocol,
    multiplicative_inverse,
)

F = TypeVar("F", bound=FieldProtocol)
T = TypeVar("T", bound=FieldProtocol)


@dataclass(init=True, frozen=True, eq=True)
class Vector(Generic[F], Iterable):
    entries: Tuple[F, ...]

    def __post_init__(self):
        for entry in self.entries:
            if not isinstance(entry, self.field):
                raise TypeError(
                    f"All entries of the vector need to be of the same type: Mismatched types: "
                    f"{type(entry)} | {self.field}"
                )

    def __len__(self) -> int:
        return len(self.entries)

    @functools.cached_property
    def field(self) -> Type:
        return type(self.entries[0])

    @classmethod
    def new_vector(
        cls, entries: Iterable[Any], field_factory: Optional[Callable[[T], F]] = None
    ) -> "Vector[F]":
        if field_factory is None:
            entries = tuple(entry for entry in entries)
        else:
            entries = tuple(field_factory(entry) for entry in entries)
        return cls(cast(Tuple[F], entries))

    def __repr__(self) -> str:
        return (
            f"abstract_algebra.modules.Vector[{self.field}]"
            f"[{','.join([entry.__repr__() for entry in self.entries])}]"
        )

    def __str__(self) -> str:
        return f"[{", ".join([entry.__str__() for entry in self.entries])}]"

    def __getitem__(self, index: int) -> F:
        return self.entries[index]

    def __iter__(self) -> Iterator[F]:
        for entry in self.entries:
            yield entry

    def convert_to(self, field_factory: Callable[[T], F]) -> "Vector[F]":
        return self.new_vector(self.entries, field_factory=field_factory)

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

    def __add__(self, other: "Vector[F]") -> "Vector[F]":
        self._validate_elementwise_operation(other=other, operator="+")
        return Vector.new_vector([x + y for x, y in zip(self.entries, other.entries)])

    def __sub__(self, other: "Vector[F]") -> "Vector[F]":
        self._validate_elementwise_operation(other=other, operator="-")
        return Vector.new_vector([x - y for x, y in zip(self.entries, other.entries)])

    def __rsub__(self, other: "Vector[F]") -> "Vector[F]":
        return other - self

    def __mul__(self, scalar: F) -> "Vector[F]":
        self._validate_scalar_operation(scalar=scalar, operator="*")
        return Vector.new_vector([x * scalar for x in self.entries])

    def __rmul__(self, scalar: F) -> "Vector[F]":
        self._validate_scalar_operation(scalar=scalar, operator="/")
        return self * scalar

    def __truediv__(self, scalar: F) -> "Vector[F]":
        return self * multiplicative_inverse(scalar)
