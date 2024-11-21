from abc import ABC, abstractmethod
from entities.vector2d import Acceleration
from entities.features import Features
from typing import Any

class BasePredictor(ABC):
    @abstractmethod
    def train(self, Any) -> Any:
        pass 

    @abstractmethod
    def predict(self, features: list[Features]) -> list[Acceleration]:
        pass