from dataclasses import dataclass
from entities.vector2d import Point2D
from abc import ABC

@dataclass(frozen=True)
class Obstacle(ABC):
    pass

@dataclass(frozen=True)
class LineObstacle(Obstacle):
    p1: Point2D
    p2: Point2D