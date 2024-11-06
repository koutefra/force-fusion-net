from dataclasses import dataclass
from entities.vector2d import Point2D, Velocity, Acceleration
from abc import ABC

@dataclass(frozen=True)
class FrameObject(ABC):
    id: int

@dataclass(frozen=True)
class PersonInFrame(FrameObject):
    position: Point2D
    velocity: Velocity
    acceleration: Acceleration