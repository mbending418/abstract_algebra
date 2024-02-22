from typing import (
    TypeVar,
    Generic,
    Tuple,
    Iterable,
    Iterator,
    Type,
    Any,
    Callable,
    cast,
    overload,
    Optional,
)
import functools
from dataclasses import dataclass
from abstract_algebra.abstract_structures.field import (
    FieldProtocol,
    multiplicative_inverse,
)
from abstract_algebra.compound_structures.vector import Vector
from abstract_algebra.linear_algebra import vector_operations

F = TypeVar("F", bound=FieldProtocol)
T = TypeVar("T", bound=FieldProtocol)
MV = TypeVar("MV", "Matrix", Vector)


@dataclass(init=True, frozen=True, eq=True)
class Matrix(Generic[F]):
    rows: Tuple[Vector[F], ...]

    def __post_init__(self):
        for row in self.rows:
            if not row.field == self.field:
                raise TypeError(
                    f"All entries of the matrix need to be of the same type: "
                    f"Mismatched types: {row.field} | {self.field}"
                )
            if len(row) != self.shape[1]:
                raise TypeError(
                    f"All rows of the matrix need to be the same dimension: "
                    f"Mismatched dims: {len(row)} | {self.shape[1]}"
                )

    @functools.cached_property
    def shape(self) -> Tuple[int, int]:
        return len(self.rows), len(self.rows[0])

    @functools.cached_property
    def field(self) -> Type:
        return type(self.rows[0][0])

    @classmethod
    def new_matrix(
        cls,
        entries: Iterable[Iterable[Any]],
        field_factory: Optional[Callable[[T], F]] = None,
    ) -> "Matrix[F]":
        entries = tuple(
            Vector.new_vector(vector_entries, field_factory)
            for vector_entries in entries
        )
        return cls(cast(Tuple[Vector[F]], entries))

    def __repr__(self) -> str:
        return f"abstract_algebra.modules.Matrix[{self.field}]{str(self)}"

    def __str__(self) -> str:
        return f"[{", ".join([row.__str__() for row in self.rows])}]"

    def __getitem__(self, index: int) -> Vector[F]:
        return self.rows[index]

    def __iter__(self) -> Iterator[Vector[F]]:
        for row in self.rows:
            yield row

    def convert_to(
        self, field_factory: Optional[Callable[[T], F]] = None
    ) -> "Matrix[F]":
        return self.new_matrix(self.rows, field_factory=field_factory)

    def _validate_scalar_operation(self, other: Any, operator: str) -> bool:
        if not isinstance(other, self.field):
            raise TypeError(
                f"unsupported operand type(s) for {operator}: "
                f"'Matrix[{self.field}]' and '{type(other)}'"
            )
        return True

    def _validate_elementwise_operation(self, other: Any, operator: str) -> bool:
        if not isinstance(other, Matrix):
            raise TypeError(
                f"unsupported operand type(s) for {operator}: 'Matrix[{self.field}]' and '{type(other)}'"
            )
        elif self.field != other.field:
            raise TypeError(
                f"unsupported operand type(s) for {operator}: "
                f"'Matrix[{self.field}]' and 'Matrix[{other.field}]'"
            )
        elif self.shape != other.shape:
            raise TypeError(
                f"unsupported operand type(s) for *: "
                f"'Matrix[{self.field}]' of size {self.shape} incompatible with"
                f"'Matrix[{other.field}]' of size {other.shape}"
            )
        return True

    def __eq__(self, other) -> bool:
        self._validate_elementwise_operation(other=other, operator="==")
        return all([x == y for x, y in zip(self.rows, other.rows)])

    def __add__(self, other: "Matrix[F]") -> "Matrix[F]":
        self._validate_elementwise_operation(other=other, operator="+")
        return Matrix.new_matrix([x + y for x, y in zip(self.rows, other.rows)])

    def __sub__(self, other: "Matrix[F]") -> "Matrix[F]":
        self._validate_elementwise_operation(other=other, operator="-")
        return Matrix.new_matrix([x - y for x, y in zip(self.rows, other.rows)])

    def __rsub__(self, other: "Matrix[F]") -> "Matrix[F]":
        return other - self

    def __mul__(self, scalar: F) -> "Matrix[F]":
        self._validate_scalar_operation(other=scalar, operator="*")
        return Matrix.new_matrix([row.__mul__(scalar) for row in self.rows])

    def __rmul__(self, scalar: F) -> "Matrix[F]":
        return self * scalar

    def __truediv__(self, scalar: F) -> "Matrix[F]":
        self._validate_scalar_operation(other=scalar, operator="/")
        return self * multiplicative_inverse(scalar)

    @overload
    def __matmul__(self, other: "Matrix[F]") -> "Matrix[F]": ...

    @overload
    def __matmul__(self, other: Vector[F]) -> Vector[F]: ...

    def __matmul__(self, other: MV) -> MV:
        if isinstance(other, Matrix):
            if self.field != other.field:
                raise TypeError(
                    f"unsupported operand type(s) for @:"
                    f"'Matrix[{self.field}]' and 'Matrix[{other.field}]'"
                )
            elif self.shape[1] != other.shape[0]:
                raise TypeError(
                    f"unsupported operand type(s) for @: "
                    f"'Matrix[{self.field}]' of size {self.shape} incompatible with"
                    f"'Matrix[{other.field}]' of size {other.shape}"
                )
            else:
                return Matrix.new_matrix(
                    [
                        [
                            vector_operations.dot_product(row, column)
                            for column in other.transpose().rows
                        ]
                        for row in self.rows
                    ]
                )
        elif isinstance(other, Vector):
            return (self @ Matrix.new_matrix([other]).transpose()).transpose()[0]
        else:
            return NotImplemented

    @overload
    def __rmatmul__(self, other: "Matrix[F]") -> "Matrix[F]": ...

    @overload
    def __rmatmul__(self, other: Vector[F]) -> Vector[F]: ...

    def __rmatmul__(self, other: MV) -> MV:
        if isinstance(other, Matrix):
            return other @ self
        elif isinstance(other, Vector):
            return (Matrix.new_matrix([other]) @ self)[0]
        else:
            return NotImplemented

    def transpose(self) -> "Matrix[F]":
        return Matrix.new_matrix(
            [[self[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])]
        )
