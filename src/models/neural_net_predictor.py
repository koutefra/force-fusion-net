import torch
from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration
from data.scene_dataset import SceneDataset
from data.torch_dataset import TorchDataset
from collections import defaultdict
from models.neural_net_model import NeuralNetModel
from entities.scene import Scene
from entities.frame import Frame
from entities.vector2d import Point2D

class NeuralNetPredictor(BasePredictor):
    def __init__(self, path: str, batch_size: int = 64, device: str | torch.device = "auto"):
        self.device = device
        super().__init__(path)
        self.model.eval()
        self.batch_size = batch_size

    def predict(self, frame: Frame, person_id: int, person_goal: Point2D) -> Acceleration:
        mock_scene = Scene(
            id=0,
            focus_person_id=person_id,
            focus_person_goal=person_goal,
            fps=float("-inf"),
            frames=[frame],
            tag=[],
            dataset="mock"
        )
        return self.predict_scene(mock_scene)[0]

    def _load_model(self, path: str) -> NeuralNetModel:
        return NeuralNetModel.from_weight_file(path, self.device)