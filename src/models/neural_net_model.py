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
        hidden_dims: list[int],
        dropout: float
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
        self.fcs_combined = self._build_mlp(combined_input_dim, hidden_dims[1:], dropout)
        self.output_layer = nn.Linear(hidden_dims[-1], self.output_dim)

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
        x_interaction = self._process_feature(x_interaction, self.fc_interaction, interaction_mask)
        x_obstacle = self._process_feature(x_obstacle, self.fc_obstacle, obstacle_mask)
        combined_features = torch.cat([x_individual, x_interaction, x_obstacle], dim=-1)
        output = self.fcs_combined(combined_features)
        output = self.output_layer(output)
        return output

    def _process_feature(self, x: torch.Tensor, layer: nn.Linear, mask: torch.Tensor) -> torch.Tensor:
        if x.numel() > 0:
            x = F.relu(layer(x))
            x = x * mask.unsqueeze(-1)
            return torch.sum(x, dim=1)
        else:
            return torch.zeros(x.shape[0], layer.out_features, device=self.device)

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
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "NeuralNetModel":
        state_dict = torch.load(path, map_location=device)
        individual_fts_dim = state_dict['fcs_combined.0.weight'].size(1) - state_dict['fc_interaction.weight'].size(0) - state_dict['fc_obstacle.weight'].size(0)
        interaction_fts_dim = state_dict['fc_interaction.weight'].size(1)
        obstacle_fts_dim = state_dict['fc_obstacle.weight'].size(1)
        
        hidden_dims = [state_dict['fc_interaction.weight'].size(0)]
        
        current_dim = state_dict['fcs_combined.0.weight'].size(0)
        hidden_dims.append(current_dim)
        layer_index = 2
        while f'fcs_combined.{layer_index}.weight' in state_dict:
            hidden_dims.append(state_dict[f'fcs_combined.{layer_index}.weight'].size(0))
            layer_index += 2  # Skipping over activation layers

        model = NeuralNetModel(
            individual_fts_dim=individual_fts_dim,
            interaction_fts_dim=interaction_fts_dim,
            obstacle_fts_dim=obstacle_fts_dim,
            hidden_dims=hidden_dims
        )

        # Load weights
        model.load_weights(path, device)
        return model

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: list[int], dropout: float) -> nn.Sequential:
        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i - 1], hidden_dim))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)