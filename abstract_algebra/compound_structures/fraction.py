from typing import Union, Generic, TypeVar, Type, Optional
from abstract_algebra.abstract_structures.monoid import additive_identity
from abstract_algebra.abstract_structures.group import additive_inverse
from abstract_algebra.abstract_structures.ring import multiplicative_identity
from abstract_algebra.abstract_structures.euclidean_ring import (
    EuclideanRingProtocol,
    generalized_gcd,
)


E = TypeVar("E", bound=EuclideanRingProtocol)


class Fraction(Generic[E]):
    numerator: E
    denominator: E
    ring: Type  # The type of E

    def __init__(self, numerator: E, denominator: Optional[E] = None):
        zero = additive_identity(numerator)
        one = multiplicative_identity(numerator)
        if denominator is None or numerator == zero:
            denominator = one
        if denominator == zero:
            raise ZeroDivisionError(
                f"Cannot set denominator to additive_identity: {zero}"
            )
        if numerator < zero and denominator < zero:
            numerator = additive_inverse(numerator)
            denominator = additive_inverse(denominator)
        q = generalized_gcd(numerator, denominator)
        self.numerator = numerator // q
        self.denominator = denominator // q
        self.ring = type(numerator)
        if not isinstance(denominator, self.ring):
            raise TypeError(
                f"numerator and denominator have to be the same type: type(numerator)={type(numerator)} | "
                f"type(denominator)={type(denominator)}"
            )

    def __str__(self) -> str:
        return f"({self.numerator})/({self.denominator})"

    def __repr__(self) -> str:
        return f"Fraction[{self.ring}]({self.numerator},{self.denominator})"

    def __eq__(self, other) -> bool:
        if isinstance(other, Fraction):
            if self.ring != other.ring:
                raise TypeError(
                    f"unsupported operand type(s) for ==: "
                    f"'Fraction[{self.ring}]' and 'Fraction[{other.ring}]'"
                )
            return (
                self.numerator * other.denominator == self.denominator * other.numerator
            )
        else:
            return NotImplemented

    def __add__(self, other: Union["Fraction[E]", E]) -> "Fraction[E]":
        if isinstance(other, Fraction):
            if self.ring != other.ring:
                raise TypeError(
                    f"unsupported operand type(s) for +: "
                    f"'Fraction[{self.ring}]' and 'Fraction[{other.ring}]'"
                )
            return Fraction(
                self.numerator * other.denominator + self.denominator * other.numerator,
                self.denominator * other.denominator,
            )
        elif isinstance(other, type(self.numerator)):
            return Fraction(self.numerator + self.denominator * other, self.denominator)
        else:
            return NotImplemented

    def __sub__(self, other: Union["Fraction[E]", E]) -> "Fraction[E]":
        if isinstance(other, Fraction):
            if self.ring != other.ring:
                raise TypeError(
                    f"unsupported operand type(s) for -: "
                    f"'Fraction[{self.ring}]' and 'Fraction[{other.ring}]'"
                )
            return Fraction(
                self.numerator * other.denominator - self.denominator * other.numerator,
                self.denominator * other.denominator,
            )
        elif isinstance(other, type(self.numerator)):
            return Fraction(self.numerator - self.denominator * other, self.denominator)
        else:
            return NotImplemented

    def __rsub__(self, other: Union["Fraction[E]", E]) -> "Fraction[E]":
        if isinstance(other, Fraction):
            return other - self
        elif isinstance(other, type(self.numerator)):
            return Fraction(other) - self
        else:
            return NotImplemented

    def __mul__(self, other: Union["Fraction[E]", E]) -> "Fraction[E]":
        if isinstance(other, Fraction):
            if self.ring != other.ring:
                raise TypeError(
                    f"unsupported operand type(s) for *: "
                    f"'Fraction[{self.ring}]' and 'Fraction[{other.ring}]'"
                )
            return Fraction(
                self.numerator * other.numerator, self.denominator * other.denominator
            )
        elif isinstance(other, type(self.numerator)):
            return Fraction(self.numerator * other, self.denominator)
        else:
            return NotImplemented

    def __rmul__(self, other: Union["Fraction[E]", E]) -> "Fraction[E]":
        return self * other

    def __truediv__(self, other: Union["Fraction[E]", E]) -> "Fraction[E]":
        if isinstance(other, Fraction):
            if self.ring != other.ring:
                raise TypeError(
                    f"unsupported operand type(s) for /: "
                    f"'Fraction[{self.ring}]' and 'Fraction[{other.ring}]'"
                )
            return Fraction(
                self.numerator * other.denominator, self.denominator * other.numerator
            )
        elif isinstance(other, type(self.numerator)):
            return Fraction(self.numerator, self.denominator * other)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Union["Fraction[E]", E]) -> "Fraction[E]":
        if isinstance(other, Fraction):
            return other / self
        elif isinstance(other, type(self.numerator)):
            return Fraction(self.denominator * other, self.numerator)
        else:
            return NotImplemented

    def get_additive_identity(self) -> "Fraction[E]":
        return Fraction(
            additive_identity(self.numerator), multiplicative_identity(self.numerator)
        )

    def get_multiplicative_identity(self) -> "Fraction":
        return Fraction(
            multiplicative_identity(self.numerator),
            multiplicative_identity(self.numerator),
        )
