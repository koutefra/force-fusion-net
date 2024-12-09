import torch
import torch.nn as nn
import torch.nn.functional as F
from entities.vector2d import kinematic_equation
from entities.batched_frames import BatchedFrames
from models.trainable_module import TrainableModule

class NeuralNetModel(TrainableModule):
    def __init__(
        self, 
        individual_fts_dim: int, 
        interaction_fts_dim: int, 
        obstacle_fts_dim: int, 
        hidden_dims: list[int] 
    ):
        super(NeuralNetModel, self).__init__()
        self.individual_fts_dim = individual_fts_dim
        self.interaction_fts_dim = interaction_fts_dim
        self.obstacle_fts_dim = obstacle_fts_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 2

        self.fc_interaction = nn.Linear(interaction_fts_dim, hidden_dims[0])
        self.fc_obstacle = nn.Linear(obstacle_fts_dim, hidden_dims[0])

        # make for hidden_dims of fcs
        combined_input_dim = individual_fts_dim + 2 * hidden_dims[1]
        self.fcs_combined = self._build_mlp(combined_input_dim, hidden_dims[1:])
        self.output_layer = nn.Linear(hidden_dims[-1], self.output_dim)

    def forward_single(
        self, 
        x_individual: torch.Tensor, 
        x_interaction: torch.Tensor,
        x_obstacle: torch.Tensor
    ) -> torch.Tensor:
        # x_individual.shape: [batch_size, individual_fts_dim] 
        # x_interaction.shape: [batch_size, l, interaction_fts_dim]
        # x_obstacle.shape: [batch_size, k, obstacle_fts_dim]
        x_interaction = self._process_feature(x_interaction, self.fc_interaction)
        x_obstacle = self._process_feature(x_obstacle, self.fc_obstacle)
        combined_features = torch.cat([x_individual, x_interaction, x_obstacle], dim=-1)
        output = self.fcs_combined(combined_features)
        output = self.output_layer(output)
        return output

    def forward(self, batched_frames: BatchedFrames, delta_times: torch.Tensor) -> torch.Tensor:
        for _ in range(batched_frames.steps_count):
            xs = batched_frames.compute_all_features()
            xs = tuple(x.to(self.device) for x in (xs if isinstance(xs, tuple) else (xs,)))
            pred_accs = self.forward_single(*xs)
            new_pos, new_vel = self.apply_predictions(
                cur_positions=batched_frames.person_positions,
                cur_velocities=batched_frames.person_velocities,
                delta_times=delta_times,
                pred_accs=pred_accs
            )
            batched_frames.update(new_pos, new_vel)
        return new_pos

    def apply_predictions(self, cur_positions, cur_velocities, delta_times, pred_accs) -> tuple[torch.Tensor, torch.Tensor]:
        next_pos_predicted, next_vel_predicted = kinematic_equation(
            cur_positions=cur_positions,
            cur_velocities=cur_velocities,
            delta_times=delta_times.unsqueeze(-1),
            cur_accelerations=pred_accs
        )
        return next_pos_predicted, next_vel_predicted

    @staticmethod
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "NeuralNetModel":
        state_dict = torch.load(path, map_location=device)

        # Infer the dimensions from the loaded state dict
        individual_fts_dim = state_dict['fc_individual.weight'].size(1)
        interaction_fts_dim = state_dict['fc_interaction.weight'].size(1)
        obstacle_fts_dim = state_dict['fc_obstacle.weight'].size(1)
        hidden_dim = state_dict['fc_individual.weight'].size(0)

        # Create the model with inferred dimensions
        model = NeuralNetModel(
            individual_fts_dim=individual_fts_dim,
            interaction_fts_dim=interaction_fts_dim,
            obstacle_fts_dim=obstacle_fts_dim,
            hidden_dim=hidden_dim
        )

        # Load weights
        model.load_weights(path, device)
        return model

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dim))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _process_feature(self, x: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        if x.numel() > 0:
            x = F.relu(layer(x))
            return torch.sum(x, dim=1)
        else:
            return torch.zeros(x.shape[0], layer.out_features, device=self.device)