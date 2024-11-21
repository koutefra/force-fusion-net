from models.base_predictor import BasePredictor
from entities.vector2d import Acceleration, Velocity, Point2D
from models.social_force_model import SocialForceModel
from entities.features import LabeledFeatures, Features
from tqdm import tqdm
from typing import Optional
import json
from itertools import product
from torch.utils.tensorboard import SummaryWriter

class SocialForcePredictor(BasePredictor):
    def __init__(self, model: SocialForceModel, logdir_path: Optional[str] = None):
        super().__init__()
        self.model = model
        self.logdir_path = logdir_path

    def train(
        self, 
        data: list[LabeledFeatures], 
        param_grid: list[dict[str, float]],
        save_path: Optional[str],
        metric_type: str = "mae"
    ) -> dict[str, float]:
        best_loss = float('inf')
        best_grid = None
        writer = SummaryWriter(self.logdir_path) if self.logdir_path else None

        with tqdm(param_grid, desc="Computing the best grid...") as progress_bar:
            for i, grid in enumerate(progress_bar):
                model = SocialForceModel(self.model.fps, **grid)
                preds_acc = model.predict(data)
                loss = [
                    self._compute_metric(
                        lf.next_pos, 
                        lf.cur_pos,
                        lf.cur_vel,
                        acc,
                        lf.delta_time,
                        metric_type=metric_type
                    )
                    for lf, acc in zip(data, preds_acc)
                ]

                avg_loss = sum(loss) / len(loss)

                progress_bar.set_description(f"Grid: {grid}, Score: {avg_loss:.4f}")
                if writer:
                    writer.add_scalar("Grid Loss", avg_loss, i)
                    writer.add_text(f"Grid {i}", f"Parameters: {json.dumps(grid)}\nLoss: {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_grid = grid

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(best_grid, f, indent=4)

        print(f"Best grid: {best_grid}, Best loss: {best_loss:.4f}")

        return grid

    @staticmethod
    def _compute_metric(
        next_pos: Point2D, 
        cur_pos: Point2D,
        cur_vel: Velocity, 
        cur_acc: Acceleration,
        delta_time: float,
        metric_type: str ="mse"
    ):
        predicted_next_pos = cur_pos + cur_vel * delta_time + 0.5 * cur_acc * delta_time**2
        error = ((predicted_next_pos.x - next_pos.x) ** 2 + (predicted_next_pos.y - next_pos.y) ** 2) ** 0.5
        if metric_type == "mse":
            return error**2
        elif metric_type == "mae":
            return error
        else:
            raise ValueError("Unknown metric type. Supported types are 'mse' and 'mae'.")

    @staticmethod
    def param_ranges_to_param_grid(param_ranges: dict[str, list[float]]) -> list[dict[str, float]]:
        return [
            dict(zip(param_ranges.keys(), values))
            for values in product(*param_ranges.values())
        ]

    def predict(self, features: list[Features]) -> list[Acceleration]:
        return self.model.predict(features)