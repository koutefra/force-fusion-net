from abc import ABC, abstractmethod
from typing import Any
from data.scene_collection import SceneCollection
from entities.vector2d import Acceleration
from entities.scene import Scene

class BasePredictor(ABC):
    def __init__(self, path: str):
        self.model = self._load_model(path)

    @abstractmethod
    def predict(self, scene_collection: SceneCollection) -> dict[int, list[Acceleration]]:
        pass
    
    @abstractmethod
    def predict_scene(self, scene: Scene) -> list[Acceleration]:
        pass

    @abstractmethod
    def _load_model(self, path: str) -> Any:
        pass