from typing import Union
import math

class Vector2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: Union['Vector2D', float, int]) -> 'Vector2D':
        if isinstance(other, Vector2D):
            return Vector2D(self.x + other.x, self.y + other.y)
        elif isinstance(other, (float, int)):
            return Vector2D(self.x + other, self.y + other)
        return NotImplemented

    def __radd__(self, other: Union[float, int]) -> 'Vector2D':
        return self + other

    def __sub__(self, other: Union['Vector2D', float, int]) -> 'Vector2D':
        if isinstance(other, Vector2D):
            return Vector2D(self.x - other.x, self.y - other.y)
        elif isinstance(other, (float, int)):
            return Vector2D(self.x - other, self.y - other)
        return NotImplemented

    def __rsub__(self, other: Union[float, int]) -> 'Vector2D':
        if isinstance(other, (float, int)):
            return Vector2D(other - self.x, other - self.y)
        return NotImplemented

    def __mul__(self, factor: Union[float, int]) -> 'Vector2D':
        return Vector2D(self.x * factor, self.y * factor)

    def __rmul__(self, factor: Union[float, int]) -> 'Vector2D':
        return self * factor

    def __truediv__(self, factor: Union[float, int]) -> 'Vector2D':
        return Vector2D(self.x / factor, self.y / factor)

    def scale(self, factor: float) -> 'Vector2D':
        """Return a scaled instance of the same type."""
        return self * factor

    def distance_to(self, other: 'Vector2D') -> float:
        """Calculate distance to another vector."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"Vector2D(x={self.x}, y={self.y})"

class Position(Vector2D):
    pass

class Velocity(Vector2D):
    pass

class Force(Vector2D):
    pass
