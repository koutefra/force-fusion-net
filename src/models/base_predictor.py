from abc import ABC, abstractmethod
from typing import Any
from data.scene_collection import SceneCollection
from entities.vector2d import Acceleration

class BasePredictor:
    def __init__(self, model: Any):
        self.model = model

    @abstractmethod
    def predict(self, scene_collection: SceneCollection) -> dict[int, list[Acceleration]]:
        pass