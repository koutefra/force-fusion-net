import torch
import os
import torchmetrics
from entities.vector2d import Acceleration, Point2D
from entities.scene import Scenes, Scene
from entities.frame import Frame, Frames
from models.base_model import BaseModel
from data.torch_dataset import TorchSceneDataset
from typing import Optional
from entities.frame import Frame

class Predictor:
    def __init__(
        self, 
        model: BaseModel,
        device: str | torch.device,
        batch_size: int = 64, 
        logdir_path: Optional[str] = None,
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
        train_data: Scenes,
        val_data: Scenes,
        pred_steps: int,
        learning_rate: float,
        epochs: int,
        loss: str = "mse"
    ) -> dict[str, torch.Tensor]:
        if loss != "mse":
            raise ValueError(f"Loss {loss} not supported")

        train_dataset = TorchSceneDataset(train_data, pred_steps, device=self.device, dtype=self.dtype) 
        eval_dataset = TorchSceneDataset(val_data, pred_steps, device=self.device, dtype=self.dtype) 

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=train_dataset.prepare_batch)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.batch_size, collate_fn=eval_dataset.prepare_batch)

        self.model.configure(
            optimizer=torch.optim.Adam(self.model.parameters(), lr=learning_rate),
            device=self.device,
            logdir=self.logdir_path,
            metrics={'MAE': torchmetrics.MeanAbsoluteError()},
            loss=torch.nn.MSELoss()
        )

        def save_model_func(self: BaseModel, epoch: int, logs: dict[str, torch.Tensor]) -> None:
            self.save_model(os.path.join(self.logdir, f'weights_run_epoch{epoch}'))
            
        logs = self.model.fit(
            train_loader, 
            epochs=epochs, 
            dev=eval_loader, 
            callbacks=[save_model_func]
        )

        return logs

    def predict(self, frame: Frame) -> dict[int, Acceleration]:
        mock_scene = Scene(
            id='mock', 
            frames=Frames({frame.number: frame}), 
            bounding_box=(Point2D.zero, Point2D.zero),
            fps=float('nan')
        )
        dataset = TorchSceneDataset(Scenes({'mock': mock_scene}), pred_steps=0, device=self.device, dtype=self.dtype) 
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.prepare_batch)
        preds_acc = self.model.predict(loader, as_numpy=True)
        preds_acc = {
            person_id: Acceleration(x=pred_acc[0], y=pred_acc[1]) 
            for pred_acc, person_id in zip(preds_acc, mock_scene.frames.to_trajectories().keys())
        }
        return preds_acc