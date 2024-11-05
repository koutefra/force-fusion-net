import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Dict, List
from models.trainable_module import TrainableModule

class NeuralNetModel(TrainableModule):
    def __init__(self, person_features_dim: int, interaction_features_dim: int, interaction_output_size: int,
                 hidden_sizes: List[int], output_size: int):
        super(NeuralNetModel, self).__init__()
        self.fc_interaction = nn.Linear(interaction_features_dim, interaction_output_size)

        self.fcs_main = nn.ModuleList()
        layer_input_size = person_features_dim + interaction_output_size
        for h_size in hidden_sizes:
            self.fcs_main.append(nn.Linear(layer_input_size, h_size))
            layer_input_size = h_size
        self.fc_output = nn.Linear(layer_input_size, output_size)

    def forward(self, x_general: torch.Tensor, x_interaction: torch.Tensor) -> torch.Tensor:
        # x_general.shape: [batch_size, general_features_dim], x_interaction.shape: [batch_size, m, interaction_features_dim]
        x_interaction_features = F.relu(self.fc_interaction(x_interaction))
        x_interaction_features_sum = torch.sum(x_interaction_features, dim=1)  # shape [batch_size, interaction_features_dim]

        x_combined = torch.concat([x_general, x_interaction_features_sum], dim=1)  # shape [batch_size, general_features_dim + interaction_features_dim]
        x = x_combined
        for layer in self.fcs_main:
            x = F.relu(layer(x))
        return self.fc_output(x)