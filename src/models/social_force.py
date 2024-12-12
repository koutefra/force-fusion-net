import torch
import json
import torch.nn as nn
from models.base_model import BaseModel
from entities.batched_frames import BatchedFrames

class SocialForce(BaseModel):
    def __init__(
        self, 
        A_interaction: float = 2.0,  # Interaction force constant, (m/s^(-2)) 
        A_obstacle: float = 6.0,  # Interaction force constant, (m/s^(-2)) 
        B_interaction: float = 0.3,  # Interaction decay constant, m
        B_obstacle: float = 0.5,  # Interaction decay constant, m
        tau: float = 0.3,  # Relaxation time constant, s
        desired_speed: float = 1.2,  # m/s
    ):
        super(SocialForce, self).__init__()
        self.A_interaction = nn.Parameter(torch.tensor(A_interaction))
        self.A_obstacle = nn.Parameter(torch.tensor(A_obstacle))
        self.B_interaction = nn.Parameter(torch.tensor(B_interaction))
        self.B_obstacle = nn.Parameter(torch.tensor(B_obstacle))
        self.tau = nn.Parameter(torch.tensor(tau))
        self.desired_speed = torch.tensor(desired_speed)

    def forward_single(
        self, 
        x_individual: torch.Tensor, 
        interaction_features: tuple[torch.Tensor, torch.Tensor],
        obstacle_features: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # x_individual.shape: [batch_size, individual_fts_dim] 
        # x_interaction.shape: [batch_size, l, interaction_fts_dim]
        # x_obstacle.shape: [batch_size, k, obstacle_fts_dim]
        x_interaction, interaction_mask = interaction_features
        x_obstacle, obstacle_mask = obstacle_features
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
        return total_force

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

    @staticmethod
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "SocialForce":
        with open(path, "r") as file:
            param_grid = json.load(file)
        model = SocialForce(**param_grid)
        model.to(device)
        return model

    def save_model(self, path: str) -> None:
        param_dict = {
            'A_interaction': self.A_interaction.item(),
            'A_obstacle': self.A_obstacle.item(),
            'B_interaction': self.B_interaction.item(),
            'B_obstacle': self.B_obstacle.item(),
            'tau': self.tau.item(),
            'desired_speed': self.desired_speed.item()
        }
        with open(path + '.json', 'w') as f:
            json.dump(param_dict, f, indent=4)