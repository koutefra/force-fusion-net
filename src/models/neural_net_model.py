import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trainable_module import TrainableModule

class NeuralNetModel(TrainableModule):
    def __init__(
        self, 
        individual_fts_dim: int, 
        interaction_fts_dim: int, 
        obstacle_fts_dim: int, 
        interaction_out_dim: int,
        obstacle_out_dim: int,
        hidden_dims: list[int], 
        output_dim: int
    ):
        super(NeuralNetModel, self).__init__()
        self.individual_fts_dim = individual_fts_dim
        self.interaction_fts_dim = interaction_fts_dim
        self.obstacle_fts_dim = obstacle_fts_dim
        self.interaction_out_dim = interaction_out_dim
        self.obstacle_out_dim = obstacle_out_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.fc_interaction = nn.Linear(interaction_fts_dim, interaction_out_dim)
        self.fc_obstacle = nn.Linear(obstacle_fts_dim, obstacle_out_dim)
        self.fcs_main = nn.ModuleList()
        layer_input_dim = individual_fts_dim + interaction_out_dim + obstacle_out_dim
        for h_dim in hidden_dims:
            self.fcs_main.append(nn.Linear(layer_input_dim, h_dim))
            layer_input_size = h_dim
        self.fc_output = nn.Linear(layer_input_size, output_dim)

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
        x_interaction = self._process_feature(x_interaction, self.fc_interaction, self.fc_interaction.out_features)
        x_obstacle = self._process_feature(x_obstacle, self.fc_obstacle, self.fc_obstacle.out_features)
        x_combined = torch.concat([x_individual, x_interaction, x_obstacle], dim=1)
        # x_combined is of shape [batch_size, individual_fts_dim + interaction_out_dim + obstacle_out_dim]
        x = x_combined
        for layer in self.fcs_main:
            x = F.relu(layer(x))
        return self.fc_output(x)

    def compute_loss(self, y_pred, y, *xs):
        """Compute the loss of the model given the inputs, predictions, and target outputs."""
        return self.loss(y_pred, y)

    def compute_metrics(self, y_pred, y, *xs, training):
        """Compute and return metrics given the inputs, predictions, and target outputs."""
        self.metrics.update(y_pred, y)
        return self.metrics.compute()

    # @staticmethod
    # def from_weight_file(path: str, device: str | torch.device = "auto") -> "NeuralNetModel":
    #     state_dict = torch.load(path, map_location="cpu")

    #     # Infer the dimensions
    #     interaction_fts_dim = state_dict['fc_interaction.weight'].size(1)
    #     interaction_output_dim = state_dict['fc_interaction.weight'].size(0)
    #     individual_fts_dim = state_dict['fcs_main.0.weight'].size(1) - interaction_output_dim

    #     hidden_dims = []
    #     for i in range(len(state_dict) - 2):  # exclude the interaction and output layer
    #         if f'fcs_main.{i}.weight' in state_dict:
    #             hidden_dims.append(state_dict[f'fcs_main.{i}.weight'].size(0))

    #     output_dim = state_dict['fc_output.weight'].size(0)

    #     model = NeuralNetModel(
    #         individual_fts_dim=individual_fts_dim,
    #         interaction_fts_dim=interaction_fts_dim,
    #         interaction_output_dim=interaction_output_dim,
    #         hidden_dims=hidden_dims,
    #         output_dim=output_dim
    #     )
    #     model.load_weights(path, device)
    #     return model