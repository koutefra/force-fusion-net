from dataclasses import dataclass
from entities.vector2d import Point2D

@dataclass(frozen=True)
class BaseObstacle:
    pass

@dataclass(frozen=True)
class PointObstacle(BaseException):
    position: Point2D 

@dataclass(frozen=True)
class LineObstacle(BaseException):
    line: tuple[Point2D, Point2D] 