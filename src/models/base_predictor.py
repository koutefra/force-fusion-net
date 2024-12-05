from abc import ABC, abstractmethod
from entities.vector2d import Acceleration
from entities.frame import Frame
from typing import Any

class BasePredictor(ABC):
    @abstractmethod
    def train(self, Any) -> Any:
        pass 

    @abstractmethod
    def predict(self, frame: Frame) -> dict[int, Acceleration]:
        pass