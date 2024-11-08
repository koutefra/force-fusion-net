from dataclasses import dataclass
from entities.vector2d import Point2D, Velocity, Acceleration
from abc import ABC

@dataclass(frozen=True)
class FrameObject(ABC):
    id: int
    position: Point2D

@dataclass(frozen=True)
class PersonInFrame(FrameObject):
    velocity: Velocity
    acceleration: Acceleration