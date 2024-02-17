from typing import (
    TypeVar,
    Generic,
    List,
    Iterable,
    Type,
    Any,
    Tuple,
    Callable,
    overload,
    Optional,
)
from functools import reduce
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.ring import multiplicative_identity
from abstract_algebra.abstract_structures.field import (
    FieldProtocol,
    multiplicative_inverse,
)
from abstract_algebra.compound_structures.vector import Vector

F = TypeVar("F", bound=FieldProtocol)
T = TypeVar("T", bound=FieldProtocol)
MV = TypeVar("MV", "Matrix", Vector)


class Matrix(Generic[F]):
    rows: List[Vector[F]]
    field: Type  # The Type of F
    shape: Tuple[int, int]

    @overload
    def __init__(self, rows: Iterable[Iterable[F]], field_factory: None = None):
        pass

    @overload
    def __init__(
        self, rows: Iterable[Iterable[Any]], field_factory: Callable[[Any], F]
    ):
        pass

    def __init__(
        self,
        rows: Iterable[Iterable[Any]],
        field_factory: Optional[Callable[[Any], F]] = None,
    ):
        self.rows: List[Vector[F]] = [
            (
                row.convert_to(field_factory=field_factory)
                if isinstance(row, Vector)
                else Vector(row, field_factory=field_factory)
            )
            for row in rows
        ]
        self.field = self.rows[0].field
        self.shape = (len(self.rows), self.rows[0].dimension)
        for row in self.rows:
            if not row.field == self.field:
                raise TypeError(
                    f"All entries of the matrix need to be of the same type: "
                    f"Mismatched types: {row.field} | {self.field}"
                )
            if row.dimension != self.shape[1]:
                raise TypeError(
                    f"All rows of the matrix need to be the same dimension: "
                    f"Mismatched dims: {row.dimension} | {self.shape[1]}"
                )

    def __repr__(self) -> str:
        return f"abstract_algebra.modules.Matrix[{self.field}]{str(self)}"

    def __str__(self) -> str:
        return f"[{", ".join([row.__str__() for row in self.rows])}]"

    def __getitem__(self, index: int) -> Vector[F]:
        return self.rows[index]

    def __setitem__(self, index: int, value: Vector[F]):
        if value.dimension != self.shape[1]:
            raise TypeError(
                "Dimension Mismatch. Cannot set a row of a "
                "'Matrix[{self.field}]' of size '{self.shape}' to: {self.value}"
            )
        self.rows[index] = value

    def convert_to(
        self, field_factory: Optional[Callable[[T], F]] = None
    ) -> "Matrix[F]":
        if field_factory is None:
            return self
        else:
            return Matrix(rows=self.rows, field_factory=field_factory)

    def zero_matrix(self) -> "Matrix[F]":
        return additive_identity(self)

    def to_vector(self) -> Vector[F]:
        if self.shape[0] == 1:
            return self.rows[0]
        elif self.shape[1] == 1:
            return self.transpose().to_vector()
        else:
            raise TypeError(
                f"Cannot convert Matrix[{self.field}] of size {self.shape} to a Vector"
            )

    def transpose(self) -> "Matrix[F]":
        return Matrix(
            [[self[i][j] for i in range(self.shape[0])] for j in range(self.shape[1])]
        )

    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    def trace(self) -> F:
        if not self.is_square():
            raise TypeError(
                f"Cannot calculate Trace of non-square matrix: shape={self.shape}"
            )
        return reduce(
            lambda a, b: a + b,
            [self[i][i] for i in range(self.shape[0])],
            additive_identity(self[0][0]),
        )

    def diagonal_product(self) -> F:
        if not self.is_square():
            raise TypeError(
                f"Cannot calculate diagonal product of non-square matrix: shape={self.shape}"
            )
        return reduce(
            lambda a, b: a * b,
            [self[i][i] for i in range(self.shape[0])],
            multiplicative_identity(self[0][0]),
        )

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
        return Matrix([x + y for x, y in zip(self.rows, other.rows)])

    def __sub__(self, other: "Matrix[F]") -> "Matrix[F]":
        self._validate_elementwise_operation(other=other, operator="-")
        return Matrix([x - y for x, y in zip(self.rows, other.rows)])

    def __rsub__(self, other: "Matrix[F]") -> "Matrix[F]":
        return other - self

    def __mul__(self, scalar: F) -> "Matrix[F]":
        self._validate_scalar_operation(other=scalar, operator="*")
        return Matrix([row.__mul__(scalar) for row in self.rows])

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
                return Matrix(
                    [
                        [row**column for column in other.transpose().rows]
                        for row in self.rows
                    ]
                )
        elif isinstance(other, Vector):
            return (self @ Matrix([other])).to_vector()
        else:
            raise TypeError(
                f"unsupported operand type(s) for @:"
                f"'Matrix[{self.field}]' and '{type(other)}'"
            )


def get_identity_matrix(dimensions: int, example_field_element: F) -> Matrix[F]:
    zero = additive_identity(example_field_element)
    one = multiplicative_identity(example_field_element)
    return Matrix(
        [
            [one if i == j else zero for j in range(dimensions)]
            for i in range(dimensions)
        ]
    )
