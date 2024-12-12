import torch
import torch.nn as nn
from models.base_model import BaseModel

class DirectNet(BaseModel):
    def __init__(
        self, 
        individual_fts_dim: int, 
        interaction_fts_dim: int, 
        obstacle_fts_dim: int, 
        hidden_dims: list[int],
        dropout: float
    ):
        super(DirectNet, self).__init__()
        self.individual_fts_dim = individual_fts_dim
        self.interaction_fts_dim = interaction_fts_dim
        self.obstacle_fts_dim = obstacle_fts_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 2
        self.dropout = dropout

        self.fc_interaction = nn.Linear(interaction_fts_dim, hidden_dims[0])
        self.fc_obstacle = nn.Linear(obstacle_fts_dim, hidden_dims[0])

        # make for hidden_dims of fcs
        combined_input_dim = individual_fts_dim + 2 * hidden_dims[0]
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

    def save_model(self, path: str) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'individual_fts_dim': self.individual_fts_dim,
            'interaction_fts_dim': self.interaction_fts_dim,
            'obstacle_fts_dim': self.obstacle_fts_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout
        }, path + '.pth')

    @staticmethod
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "DirectNet":
        checkpoint = torch.load(path, map_location=device)
        individual_fts_dim = checkpoint['individual_fts_dim']
        interaction_fts_dim = checkpoint['interaction_fts_dim']
        obstacle_fts_dim = checkpoint['obstacle_fts_dim']
        hidden_dims = checkpoint['hidden_dims']
        dropout = checkpoint['dropout']

        model = DirectNet(
            individual_fts_dim=individual_fts_dim,
            interaction_fts_dim=interaction_fts_dim,
            obstacle_fts_dim=obstacle_fts_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model