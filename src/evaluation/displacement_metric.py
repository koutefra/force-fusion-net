import torch
import torchmetrics
from typing import Dict

class DisplacementMetric(torchmetrics.Metric):
    def __init__(self, steps_update_period: int):
        super().__init__()
        self.steps_update_period = steps_update_period
        self.add_state("predictions", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("metric_results_accumulated", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("steps_accumulated", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, data_id: Dict[str, int]):
        self.predictions = torch.cat([self.predictions, preds.detach().unsqueeze(0)])
        self.targets = torch.cat([self.targets, target.detach().unsqueeze(0)])
        self.steps_accumulated += 1

        scene_id, frame_id, person_id = data_id["scene_id"], data_id["frame_id"], data_id["person_id"]

        print(data_id)

        if self.steps_accumulated >= self.steps_update_period:
            metric_result = self._compute_metric(self.predictions, self.targets)
            self.metric_results_accumulated = torch.cat([self.metric_results_accumulated, metric_result.unsqueeze(0)])
            self.predictions = torch.tensor([])
            self.targets = torch.tensor([])
            self.steps_accumulated = torch.tensor(0)

    def compute(self):
        return self.metric_results_accumulated.mean()

    def _compute_metric(self, preds, targets):
        print(preds.shape, targets.shape)
        return (preds - targets).abs().mean()