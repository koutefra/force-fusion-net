from entities.scene import Scene
from entities.vector2d import Point2D
from models.base_model import BaseModel
from models.predictor import Predictor
from typing import Callable, Any, Optional
import torch
from torch import Tensor
from evaluation.frame_evaluator import FrameEvaluator
import torchmetrics
from torchmetrics.functional.regression.mae import _mean_absolute_error_update

class Evaluator:
    class OneStepMAE(torchmetrics.MeanAbsoluteError):
        def __init__(self, step_dim_id: int, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.dim = step_dim_id

        def update(self, preds: Tensor, target: Tensor) -> None:
            sum_abs_error, num_obs = _mean_absolute_error_update(preds[:, self.dim, :], target[:, self.dim, :])
            self.sum_abs_error += sum_abs_error
            self.total += num_obs

    def _max_error_update(preds: Tensor, target: Tensor) -> Tensor:
        """Compute the element-wise maximum error."""
        return torch.max(torch.abs(preds - target))

    class OneStepMaxError(torchmetrics.Metric):
        def __init__(self, step_dim_id: int, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.dim = step_dim_id
            self.add_state("max_error", default=torch.tensor(0.0), dist_reduce_fx="max")

        def update(self, preds: Tensor, target: Tensor) -> None:
            """Update the state with new predictions and targets."""
            current_max_error = torch.max(torch.abs(preds[:, self.dim, :] - target[:, self.dim, :]))
            self.max_error = torch.max(self.max_error, current_max_error)

        def compute(self) -> Tensor:
            """Return the maximum error."""
            return self.max_error

    def __init__(self):
        self.frame_evaluator = FrameEvaluator() 

    def evaluate_scene(self, scene: Scene, agent_coll_thr: float = 0.14, obstacle_coll_thr: float = 0.05) -> dict[str, float]:
        return self.frame_evaluator.evaluate_frames(
            scene.frames, 
            agent_coll_thr=agent_coll_thr,
            obstacle_coll_thr=obstacle_coll_thr,
        )