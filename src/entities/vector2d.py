import math
from dataclasses import dataclass
from typing import Union

@dataclass(frozen=True)
class Point2D:
    x: float = 0.0
    y: float = 0.0

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> tuple[int, int]:
        return (int(self.x), int(self.y))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.x}, {self.y})"

    def __add__(self, other: Union[int, float, "Point2D"]) -> "Point2D":
        if isinstance(other, Point2D):
            return self.__class__(self.x + other.x, self.y + other.y)
        elif isinstance(other, (int, float)):
            return self.__class__(self.x + other, self.y + other)
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __sub__(self, other: Union[int, float, "Point2D"]) -> "Point2D":
        return self + (-other) 

    def __mul__(self, scalar: float) -> "Point2D":
        return self.__class__(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Point2D":
        return self.__mul__(scalar)

    def __neg__(self) -> "Point2D":
        return self.__class__(-self.x, -self.y)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2)

    def normalize(self) -> "Point2D":
        mag = self.magnitude()
        if mag == 0:
            return self.__class__(0, 0)
        return self.__class__(self.x / mag, self.y / mag)

    def dot(self, other: "Point2D") -> float:
        return self.x * other.x + self.y * other.y

    def direction_to(self, other: "Point2D") -> "Point2D":
        return (other - self).normalize()

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def clamp(self, max_magnitude: float) -> "Point2D":
        mag = self.magnitude()
        if mag > max_magnitude:
            return self.normalize() * max_magnitude
        return self

    @staticmethod
    def zero() -> "Point2D":
        return Point2D(0.0, 0.0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point2D):
            return False
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __getitem__(self, index: int) -> float:
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            raise IndexError("Index out of range for Point2D")

    def __len__(self) -> int:
        return 2

@dataclass(frozen=True)
class Velocity(Point2D):
    @staticmethod
    def from_points(pos1: Point2D, pos2: Point2D, delta_time: float) -> "Velocity":
        dx = (pos2.x - pos1.x) / delta_time
        dy = (pos2.y - pos1.y) / delta_time
        return Velocity(dx, dy)

@dataclass(frozen=True)
class Acceleration(Point2D):
    @staticmethod
    def from_velocities(vel1: Velocity, vel2: Velocity, delta_time: float) -> "Acceleration":
        dvx = (vel2.x - vel1.x) / delta_time
        dvy = (vel2.y - vel1.y) / delta_time
        return Acceleration(dvx, dvy)