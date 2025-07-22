from entities.scene import Scene
from entities.vector2d import Point2D
from models.base_model import BaseModel
from models.predictor import Predictor
from typing import Callable, Any, Optional
import torch
from torch import Tensor
import numpy as np
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

    def evaluate_scene(self, scene: Scene, agent_coll_thr: float = 0.2, obstacle_coll_thr: float = 0.07) -> dict[str, float]:
        return self.frame_evaluator.evaluate_frames(
            scene.frames, 
            agent_coll_thr=agent_coll_thr,
            obstacle_coll_thr=obstacle_coll_thr,
        )

    def evaluate_force_magnitudes(self, scene: Scene) -> dict[str, float]:
        magnitudes = [
            p.acceleration.magnitude()
            for frame in scene.frames.values()
            for p in frame.persons.values()
            if p.acceleration is not None
        ]
        return {
            "force_mean": float(np.mean(magnitudes)) if magnitudes else 0.0,
            "force_std": float(np.std(magnitudes)) if magnitudes else 0.0,
            "force_max": float(np.max(magnitudes)) if magnitudes else 0.0,
        }

    def evaluate_force_deviation(self, scene_a: Scene, scene_b: Scene) -> dict[str, float]:
        from scipy.spatial.distance import cosine
        l2s, coss = [], []
        for t in scene_a.frames:
            frame_a, frame_b = scene_a.frames[t], scene_b.frames.get(t)
            if not frame_b: continue
            for pid in frame_a.persons:
                acc_a = frame_a.persons[pid].acceleration
                acc_b = frame_b.persons.get(pid, None)
                if acc_a and acc_b and acc_b.acceleration:
                    v1 = np.array([acc_a.x, acc_a.y])
                    v2 = np.array([acc_b.acceleration.x, acc_b.acceleration.y])
                    l2s.append(np.linalg.norm(v1 - v2))
                    coss.append(cosine(v1, v2))
        return {
            "force_L2": float(np.mean(l2s)) if l2s else 0.0,
            "force_cosine": float(np.mean(coss)) if coss else 0.0,
        }

    def evaluate_individual_ADE_FDE(self, scene_gt: Scene, scene_pred: Scene) -> dict[str, float]:
        ped_errors = []
        for pid in scene_gt.get_all_person_ids():
            gt_traj = scene_gt.get_person_trajectory(pid)
            pred_traj = scene_pred.get_person_trajectory(pid)

            if not gt_traj or not pred_traj:
                continue  # skip if one is missing

            min_len = min(len(gt_traj), len(pred_traj))
            gt_traj = gt_traj[:min_len]
            pred_traj = pred_traj[:min_len]

            dists = [np.linalg.norm(gt - pred) for gt, pred in zip(gt_traj, pred_traj)]
            if dists:
                ped_errors.append((np.mean(dists), dists[-1]))

        if not ped_errors:
            return {"ADE_ped_mean": 0.0, "ADE_ped_std": 0.0, "FDE_ped_mean": 0.0, "FDE_ped_std": 0.0}

        ades, fdes = zip(*ped_errors)
        return {
            "ADE_ped_mean": float(np.mean(ades)),
            "ADE_ped_std": float(np.std(ades)),
            "FDE_ped_mean": float(np.mean(fdes)),
            "FDE_ped_std": float(np.std(fdes)),
        }

    def evaluate_min_distances(self, scene: Scene) -> dict[str, float]:
        min_dists = []
        for frame in scene.frames.values():
            persons = list(frame.persons.values())
            for i, p1 in enumerate(persons):
                dists = [
                    np.linalg.norm(p1.position - p2.position)
                    for j, p2 in enumerate(persons) if i != j
                ]
                if dists:
                    min_dists.append(min(dists))
        return {
            "min_dist_mean": float(np.mean(min_dists)) if min_dists else 0.0,
            "min_dist_std": float(np.std(min_dists)) if min_dists else 0.0,
            "min_dist_min": float(np.min(min_dists)) if min_dists else 0.0,
        }
