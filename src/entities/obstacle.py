from dataclasses import dataclass
from entities.vector2d import Point2D
from abc import ABC, abstractmethod
from typing import Callable

@dataclass(frozen=True)
class Obstacle(ABC):
    @abstractmethod
    def normalized(self) -> "Obstacle":
        pass

@dataclass(frozen=True)
class LineObstacle(Obstacle):
    p1: Point2D
    p2: Point2D

    def normalized(self, pos_scale: Callable[[Point2D], Point2D] = lambda x: x) -> "LineObstacle":
        return LineObstacle(
            p1=pos_scale(self.p1),
            p2=pos_scale(self.p2)
        )