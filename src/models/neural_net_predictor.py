import torch
import torchmetrics
from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration
from models.neural_net_model import NeuralNetModel
from entities.features import Features
from data.torch_dataset import TorchSceneDataset
from typing import Optional

class NeuralNetPredictor(BasePredictor):
    def __init__(
        self, 
        model: NeuralNetModel,
        batch_size: int = 64, 
        logdir_path: Optional[str] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.batch_size = batch_size
        self.logdir_path = logdir_path
        self.device = device
        self.dtype = dtype

    def train(
        self,
        train_features: list[Features],
        val_features: list[Features],
        learning_rate: float,
        epochs: int,
        save_path: Optional[str],
        loss: str = "mse"
    ) -> None:
        if loss != "mse":
            raise ValueError(f"Loss {loss} not supported")

        train_dataset = TorchSceneDataset(train_features, device=self.device, dtype=self.dtype) 
        eval_dataset = TorchSceneDataset(val_features, device=self.device, dtype=self.dtype) 

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, collate_fn=train_dataset.prepare_batch)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.batch_size, collate_fn=eval_dataset.prepare_batch)

        self.model.configure(
            optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate),
            device=self.device,
            logdir=self.logdir_path,
            metrics={'MAE': torchmetrics.MeanAbsoluteError()},
            loss=torch.nn.MSELoss()
        )

        logs = self.model.fit(train_loader, dev=eval_loader, epochs=epochs, callbacks=[])
        if save_path:
            self.model.save_weights(save_path)

    def predict(self, features: list[Features]) -> list[Acceleration]:
        features_ts = [f.to_labeled_features() for f in features]
        dataset = TorchSceneDataset(features_ts, device=self.device, dtype=self.dtype) 
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.prepare_batch)
        preds_acc = self.model.predict(loader, as_numpy=True)
        preds_acc = [Acceleration(x=pred_acc[0], y=pred_acc[1]) for pred_acc in preds_acc]
        return preds_acc