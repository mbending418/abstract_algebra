from typing import Union
from dataclasses import dataclass
from abstract_algebra.abstract_structures.euclidean_ring import EuclideanRingProtocol
from abstract_algebra.abstract_structures.field import FieldProtocol


@dataclass(init=True, frozen=True)
class ComplexNumber(FieldProtocol):
    real: float
    imaginary: float = 0.0

    def __eq__(self, other) -> bool:
        if isinstance(other, ComplexNumber):
            return (self.real == other.real) and (self.imaginary == other.imaginary)
        if isinstance(other, (int, float)):
            return (self.imaginary == 0) and (self.real == other)
        else:
            return False

    def __add__(self, other: Union["ComplexNumber", int, float]) -> "ComplexNumber":
        if isinstance(other, int):
            other = ComplexNumber(other)
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                self.real + other.real, self.imaginary + other.imaginary
            )
        else:
            return NotImplemented

    def __radd__(self, other: Union["ComplexNumber", int, float]) -> "ComplexNumber":
        return self + other

    def __sub__(self, other: Union["ComplexNumber", int, float]) -> "ComplexNumber":
        if isinstance(other, (int, float)):
            other = ComplexNumber(other)
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                self.real - other.real, self.imaginary - other.imaginary
            )
        else:
            return NotImplemented

    def __rsub__(self, other: Union["ComplexNumber", int, float]) -> "ComplexNumber":
        if isinstance(other, (int, float)):
            other = ComplexNumber(other)
        if isinstance(other, ComplexNumber):
            return other - self
        else:
            return NotImplemented

    def __mul__(self, other: Union["ComplexNumber", int, float]) -> "ComplexNumber":
        if isinstance(other, (int, float)):
            other = ComplexNumber(other)
        if isinstance(other, ComplexNumber):
            return ComplexNumber(
                self.real * other.real - self.imaginary * other.imaginary,
                self.real * other.imaginary + self.imaginary * other.real,
            )
        else:
            return NotImplemented

    def __rmul__(self, other: Union["ComplexNumber", int, float]) -> "ComplexNumber":
        return self * other

    def __truediv__(self, other: Union["ComplexNumber", int, float]) -> "ComplexNumber":
        if isinstance(other, (int, float)):
            other = ComplexNumber(other)
        if isinstance(other, ComplexNumber):
            denominator = other.real**2 + other.imaginary**2
            return ComplexNumber(
                (self.real * other.real + self.imaginary * other.imaginary)
                / denominator,
                (self.imaginary * other.real - self.real * other.imaginary)
                / denominator,
            )
        else:
            return NotImplemented

    def get_additive_identity(self) -> "ComplexNumber":
        return ComplexNumber(0.0)

    def get_multiplicative_identity(self) -> "ComplexNumber":
        return ComplexNumber(1.0)


@dataclass(init=True, frozen=True)
class GaussianInteger(EuclideanRingProtocol):
    real: int
    imaginary: int = 0

    def norm2(self) -> int:
        return self.real**2 + self.imaginary**2

    def __eq__(self, other) -> bool:
        if isinstance(other, GaussianInteger):
            return (self.real == other.real) and (self.imaginary == other.imaginary)
        else:
            return NotImplemented

    def __gt__(self, other: Union["GaussianInteger", int]) -> bool:
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            return self.norm2() > other.norm2()
        else:
            return NotImplemented

    def __lt__(self, other: Union["GaussianInteger", int]) -> bool:
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            return self.norm2() < other.norm2()
        else:
            return NotImplemented

    def __add__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            return GaussianInteger(
                self.real + other.real, self.imaginary + other.imaginary
            )
        else:
            return NotImplemented

    def __radd__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        return self + other

    def __sub__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            return GaussianInteger(
                self.real - other.real, self.imaginary - other.imaginary
            )
        else:
            return NotImplemented

    def __rsub__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            return other - self
        else:
            return NotImplemented

    def __mul__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            return GaussianInteger(
                self.real * other.real - self.imaginary * other.imaginary,
                self.real * other.imaginary + self.imaginary * other.real,
            )
        else:
            return NotImplemented

    def __rmul__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        return self * other

    def __floordiv__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            norm = other.norm2()
            real_num = self.real * other.real + self.imaginary * other.imaginary
            imaginary_num = self.imaginary * other.real - self.real * other.imaginary
            numerator = GaussianInteger(real_num, imaginary_num)
            base_candidate = GaussianInteger(real_num // norm, imaginary_num // norm)
            one = GaussianInteger(1)
            i = GaussianInteger(0, 1)
            candidates = [
                base_candidate - i - one,
                base_candidate - i,
                base_candidate - i + one,
                base_candidate - one,
                base_candidate,
                base_candidate + one,
                base_candidate + i - one,
                base_candidate + i,
                base_candidate + i + one,
            ]
            winning_candidate = base_candidate
            for candidate in candidates:
                if (numerator - candidate * norm).norm2() < (
                    numerator - winning_candidate * norm
                ).norm2():
                    winning_candidate = candidate
            return winning_candidate
        else:
            return NotImplemented

    def __rfloordiv__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        if isinstance(other, int):
            other = GaussianInteger(other)
        if isinstance(other, GaussianInteger):
            return other - self
        else:
            return NotImplemented

    def __mod__(self, other: Union["GaussianInteger", int]) -> "GaussianInteger":
        if isinstance(other, (int, GaussianInteger)):
            return self - (self // other) * other

    def get_additive_identity(self) -> "GaussianInteger":
        return GaussianInteger(0)

    def get_multiplicative_identity(self) -> "GaussianInteger":
        return GaussianInteger(1)
