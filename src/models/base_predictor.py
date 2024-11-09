from abc import ABC, abstractmethod
from typing import Any
from data.scene_collection import SceneCollection
from entities.vector2d import Acceleration, Point2D
from entities.scene import Scene
from entities.frame import Frame

class BasePredictor(ABC):
    def __init__(self, path: str):
        self.model = self._load_model(path)

    @abstractmethod
    def predict_frame(self, frame: Frame, person_id: int, person_goal: Point2D) -> Acceleration:
        pass

    @abstractmethod
    def _load_model(self, path: str) -> Any:
        pass

    def predict(self, scene_collection: SceneCollection) -> dict[int, list[Acceleration]]:
        preds = {}
        for scene_id, scene in scene_collection.scenes.items():
            scene_preds = self.predict_scene(scene)
            preds[scene_id] = scene_preds
        return preds
    
    def predict_scene(self, scene: Scene) -> list[Acceleration]:
        preds = []
        for frame in scene.frames:
            frame_pred = self.predict_frame(frame, scene.focus_person_id)
            preds.append(frame_pred)
        return preds