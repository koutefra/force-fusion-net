from abc import ABC, abstractmethod
from entities.vector2d import Acceleration
from entities.features import Features

class BasePredictor(ABC):
    @abstractmethod
    def predict(self, features: list[Features]) -> list[Acceleration]:
        pass