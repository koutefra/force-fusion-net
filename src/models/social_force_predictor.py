from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration
from data.scene_collection import SceneCollection
from models.social_force_model import SocialForceModel

class SocialForcePredictor(BasePredictor):
    def __init__(self, model: SocialForceModel):
        super().__init__(model)

    def predict(self, scene_collection: SceneCollection) -> dict[int, list[Acceleration]]:
        return self.model.predict_scenes(scene_collection.scenes)