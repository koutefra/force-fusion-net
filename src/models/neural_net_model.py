import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trainable_module import TrainableModule

class NeuralNetModel(TrainableModule):
    def __init__(
        self, 
        individual_fts_dim: int, 
        interaction_fts_dim: int, 
        interaction_output_dim: int,
        hidden_dims: list[int], 
        output_dim: int
    ):
        super(NeuralNetModel, self).__init__()
        self.individual_fts_dim = individual_fts_dim
        self.interaction_fts_dim = interaction_fts_dim
        self.interaction_output_dim = interaction_output_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.fc_interaction = nn.Linear(interaction_fts_dim, interaction_output_dim)
        self.fcs_main = nn.ModuleList()
        layer_input_dim = individual_fts_dim + interaction_output_dim
        for h_dim in hidden_dims:
            self.fcs_main.append(nn.Linear(layer_input_dim, h_dim))
            layer_input_size = h_dim
        self.fc_output = nn.Linear(layer_input_size, output_dim)

    def forward(self, x_individual: torch.Tensor, x_interaction: torch.Tensor) -> torch.Tensor:
        # x_individual.shape: [batch_size, individual_fts_dim] 
        # x_interaction.shape: [batch_size, m, interaction_fts_dim]

        if x_interaction.numel() != 0: # not empty
            x_interaction = self.fc_interaction(x_interaction)
            x_interaction = F.relu(x_interaction)
            x_interaction= torch.sum(x_interaction, dim=1)
        else:
            x_interaction = torch.zeros(x_interaction.shape[0],  self.interaction_output_dim)

        x_combined = torch.concat([x_individual, x_interaction], dim=1)
        # x_combined is of shape [batch_size, individual_fts_dim + interaction_output_dim]
        x = x_combined
        for layer in self.fcs_main:
            x = F.relu(layer(x))
        return self.fc_output(x)