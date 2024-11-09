import json
from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration, Point2D
from data.scene_collection import SceneCollection
from models.social_force_model import SocialForceModel
from entities.scene import Scene
from entities.frame import Frame

class SocialForcePredictor(BasePredictor):
    def __init__(self, path: str):
        super().__init__(path)

    def predict_frame(self, frame: Frame, person_id: int, person_goal: Point2D) -> Acceleration:
        return self.model.predict_frame(frame, person_id, person_goal)

    def predict(self, scene_collection: SceneCollection) -> dict[int, list[Acceleration]]:
        return self.model.predict_scenes(scene_collection.scenes)

    def predict_scene(self, scene: Scene) -> list[Acceleration]:
        return self.model.predict_scene(scene)

    def _load_model(self, path: str) -> SocialForceModel:
        with open(path, 'r') as file:
            params = json.load(file)
        return SocialForceModel(**params)