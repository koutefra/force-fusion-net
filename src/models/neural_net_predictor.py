import torch
from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration
from models.neural_net_model import NeuralNetModel
from entities.features import Features
from data.torch_dataset import TorchSceneDataset

class NeuralNetPredictor(BasePredictor):
    def __init__(
        self, 
        model: NeuralNetModel,
        batch_size: int = 64, 
        device: str | torch.device = "cpu",
        dtype = torch.float32
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

    def predict(self, features: list[Features]) -> list[Acceleration]:
        features_ts = [f.to_labeled_features().to_tensor(self.device, self.dtype) for f in features]
        dataset = TorchSceneDataset(features_ts, device=self.device, dtype=self.dtype) 
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.prepare_batch)
        preds_acc = self.model.predict(loader, as_numpy=True)
        return preds_acc