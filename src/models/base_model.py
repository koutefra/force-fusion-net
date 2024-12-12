import torch
from entities.vector2d import kinematic_equation
from entities.batched_frames import BatchedFrames
from models.trainable_module import TrainableModule
from abc import ABC, abstractmethod

class BaseModel(TrainableModule, ABC):
    @abstractmethod
    def forward_single(
        self, 
        x_individual: torch.Tensor, 
        interaction_features: tuple[torch.Tensor, torch.Tensor],
        obstacle_features: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        pass

    def forward(self, batched_frames: BatchedFrames) -> torch.Tensor:
        pred_next_pos_all_steps = []
        for step in range(batched_frames.steps_count):
            features = batched_frames.compute_all_features()
            pred_accs = self.forward_single(*features)
            new_pos, new_vel = self.apply_predictions(
                cur_positions=batched_frames.person_positions,
                cur_velocities=batched_frames.person_velocities,
                delta_times=batched_frames.get_delta_times(),
                pred_accs=pred_accs
            )
            pred_next_pos_all_steps.append(new_pos)

            if step != batched_frames.steps_count - 1:  # not last step
                batched_frames.update(new_pos, new_vel)

        return torch.stack(pred_next_pos_all_steps, dim=1)

    def apply_predictions(self, cur_positions, cur_velocities, delta_times, pred_accs) -> tuple[torch.Tensor, torch.Tensor]:
        next_pos_predicted, next_vel_predicted = kinematic_equation(
            cur_positions=cur_positions,
            cur_velocities=cur_velocities,
            delta_times=delta_times.unsqueeze(-1),
            cur_accelerations=pred_accs
        )
        return next_pos_predicted, next_vel_predicted

    def predict_step(self, xs, as_numpy=True):
        """An overridable method performing a single prediction step."""
        with torch.no_grad():
            batched_frames = xs
            features = batched_frames.compute_all_features()
            batch = self.forward_single(*features)
            if type(batch) is list:
                batch = torch.stack(batch)
            return batch.numpy(force=True) if as_numpy else batch

    @staticmethod
    @abstractmethod
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "BaseModel":
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        pass