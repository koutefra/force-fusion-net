from abc import ABC, abstractmethod
from typing import Any
from data.scene_dataset import SceneDataset
from entities.vector2d import Acceleration, Point2D
from entities.scene import Frame
from entities.obstacle import BaseObstacle

class BasePredictor(ABC):
    def __init__(self, path: str):
        self.model = self._load_model(path)

    @abstractmethod
    def predict(self, frame: Frame, obstacles: list[BaseObstacle]) -> Acceleration:
        pass

    @abstractmethod
    def _load_model(self, path: str) -> Any:
        pass