import json
from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration, Point2D
from data.scene_dataset import SceneDataset
from models.social_force_model import SocialForceModel
from entities.scene import Scene
from entities.frame import Frame

class SocialForcePredictor(BasePredictor):
    def __init__(self, path: str):
        super().__init__(path)

    def predict(self, frame: Frame, person_id: int, person_goal: Point2D) -> Acceleration:
        return self.model.predict(frame, person_id, person_goal)

    def _load_model(self, path: str) -> SocialForceModel:
        with open(path, 'r') as file:
            params = json.load(file)
        return SocialForceModel(**params)