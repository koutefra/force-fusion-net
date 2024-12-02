import torch
import torch.nn as nn
import torch.nn.functional as F
from entities.vector2d import kinematic_equation
from models.trainable_module import TrainableModule

class NeuralNetModel(TrainableModule):
    def __init__(
        self, 
        individual_fts_dim: int, 
        interaction_fts_dim: int, 
        obstacle_fts_dim: int, 
        hidden_dim: int 
    ):
        super(NeuralNetModel, self).__init__()
        self.individual_fts_dim = individual_fts_dim
        self.interaction_fts_dim = interaction_fts_dim
        self.obstacle_fts_dim = obstacle_fts_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 2

        #  individual features
        self.b_norm_individual = nn.BatchNorm1d(individual_fts_dim)
        self.fc_individual = nn.Linear(individual_fts_dim, hidden_dim)

        #  interaction features
        self.b_norm_interaction = nn.BatchNorm1d(interaction_fts_dim)
        self.fc_interaction = nn.Linear(interaction_fts_dim, hidden_dim)

        #  obstacle features
        self.b_norm_obstacle = nn.BatchNorm1d(obstacle_fts_dim)
        self.fc_obstacle = nn.Linear(obstacle_fts_dim, hidden_dim)

        self.fc_combined = nn.Linear(3 * hidden_dim, self.output_dim)

    def _process_feature(self, x: torch.Tensor, layer: nn.Linear, out_dim: int) -> torch.Tensor:
        if x.numel() > 0:
            x = F.relu(layer(x))
            return torch.sum(x, dim=1)
        else:
            return torch.zeros(x.shape[0], out_dim, device=self.device)

    def forward(
        self, 
        x_individual: torch.Tensor, 
        x_interaction: torch.Tensor,
        x_obstacle: torch.Tensor
    ) -> torch.Tensor:
        # x_individual.shape: [batch_size, individual_fts_dim] 
        # x_interaction.shape: [batch_size, l, interaction_fts_dim]
        # x_obstacle.shape: [batch_size, k, obstacle_fts_dim]

        x_individual = self.b_norm_individual(x_individual)
        x_individual = self.fc_individual(x_individual)
        x_individual = F.relu(x_individual)

        if x_interaction.numel() > 0:
            orig_shape = x_interaction.shape
            x_interaction = x_interaction.view(-1, x_interaction.size(-1))
            x_interaction = self.b_norm_interaction(x_interaction)
            x_interaction = self.fc_interaction(x_interaction)
            x_interaction = F.relu(x_interaction)
            x_interaction = x_interaction.view(*orig_shape[:-1], -1).sum(dim=1)
        else:
            x_interaction = torch.zeros_like(x_individual)

        if x_obstacle.numel() > 0:
            orig_shape = x_obstacle.shape
            x_obstacle = x_obstacle.view(-1, x_obstacle.size(-1))
            x_obstacle = self.b_norm_obstacle(x_obstacle)
            x_obstacle = self.fc_obstacle(x_obstacle)
            x_obstacle = F.relu(x_obstacle)
            x_obstacle = x_obstacle.view(*orig_shape[:-1], -1).sum(dim=1)
        else:
            x_obstacle = torch.zeros_like(x_individual)

        combined_features = torch.cat([x_individual, x_interaction, x_obstacle], dim=-1)
        output = self.fc_combined(combined_features)

        return output

    def compute_loss(self, y_pred, ys, *xs):
        next_pos_predicted = kinematic_equation(
            cur_positions=ys[0],
            cur_velocities=ys[2],
            delta_times=ys[3].unsqueeze(-1),
            cur_accelerations=y_pred
        )
        next_positions = ys[1]
        return self.loss(next_pos_predicted, next_positions)

    def compute_metrics(self, y_pred, ys, *xs, training):
        next_pos_predicted = kinematic_equation(
            cur_positions=ys[0],
            cur_velocities=ys[2],
            delta_times=ys[3].unsqueeze(-1),
            cur_accelerations=y_pred
        )
        next_positions = ys[1]
        self.metrics.update(next_pos_predicted, next_positions)
        return self.metrics.compute()

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