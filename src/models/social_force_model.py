from entities.batched_frames import BatchedFrames
import numpy as np
import torch

class SocialForceModel:
    def __init__(
        self, 
        A_interaction: float = 2.0,  # Interaction force constant, (m/s^(-2)) 
        A_obstacle: float = 6.0,  # Interaction force constant, (m/s^(-2)) 
        B_interaction: float = 0.3,  # Interaction decay constant, m
        B_obstacle: float = 0.5,  # Interaction decay constant, m
        tau: float = 0.3,  # Relaxation time constant, s
        desired_speed: float = 0.8,  # m/s
    ):
        self.A_interaction = A_interaction
        self.A_obstacle = A_obstacle
        self.B_interaction = B_interaction
        self.B_obstacle = B_obstacle
        self.tau = tau
        self.desired_speed = desired_speed

    def _desired_force(
        self, 
        dir_x_to_goal: torch.Tensor, 
        dir_y_to_goal: torch.Tensor,
        velocity_x: torch.Tensor,
        velocity_y: torch.Tensor
    ) -> torch.Tensor:
        dir_to_goal = torch.stack((dir_x_to_goal, dir_y_to_goal), dim=-1)
        norm = torch.norm(dir_to_goal, dim=-1, keepdim=True)
        normalized_dir_to_goal = dir_to_goal / norm
        desired_velocity = normalized_dir_to_goal * self.desired_speed
        velocity = torch.stack((velocity_x, velocity_y), dim=-1)
        desired_acceleration = (desired_velocity - velocity) / self.tau
        return desired_acceleration  # a tensor of shape [n, 2]

    def _compute_force(
        self,
        direction_x: torch.Tensor,
        direction_y: torch.Tensor,
        distance: torch.Tensor,
        A: float, 
        B: float,
        mask: torch.Tensor
    ) -> torch.Tensor:
        force_magnitude = A * torch.exp(-distance / B)
        force_magnitude *= mask
        total_force_x = -torch.sum(direction_x * force_magnitude, dim=1)
        total_force_y = -torch.sum(direction_y * force_magnitude, dim=1)
        return torch.stack((total_force_x, total_force_y), dim=-1)

    def predict(self, batched_frames: BatchedFrames, as_numpy: bool = True) -> torch.Tensor | np.ndarray: 
        if batched_frames.steps_count != 1:
            raise ValueError("Social force model does not allow multiple steps ahead prediction.")
        features = batched_frames.compute_all_features()
        x_individual, (x_interaction, interaction_mask), (x_obstacle, obstacle_mask) = features
        desired_force = self._desired_force(
            x_individual[:, BatchedFrames.get_individual_feature_index('goal_dir_x')],
            x_individual[:, BatchedFrames.get_individual_feature_index('goal_dir_y')],
            x_individual[:, BatchedFrames.get_individual_feature_index('vel_x')],
            x_individual[:, BatchedFrames.get_individual_feature_index('vel_y')]
        )
        interaction_force = self._compute_force(
            x_interaction[:, :, BatchedFrames.get_interaction_feature_index('dir_x')],
            x_interaction[:, :, BatchedFrames.get_interaction_feature_index('dir_y')],
            x_interaction[:, :, BatchedFrames.get_interaction_feature_index('dist')],
            self.A_interaction,
            self.B_interaction,
            interaction_mask
        )
        obstacle_force = self._compute_force(
            x_obstacle[:, :, BatchedFrames.get_obstacle_feature_index('dir_closest_x')],
            x_obstacle[:, :, BatchedFrames.get_obstacle_feature_index('dir_closest_y')],
            x_obstacle[:, :, BatchedFrames.get_obstacle_feature_index('dist_closest')],
            self.A_obstacle,
            self.B_obstacle,
            obstacle_mask
        )
        total_force = desired_force + interaction_force + obstacle_force
        return total_force.numpy(force=True) if as_numpy else total_force