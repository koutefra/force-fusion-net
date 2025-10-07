import torch
import torch.nn as nn
from models.base_model import BaseModel

class FusionNet(BaseModel):
    def __init__(
        self, 
        individual_fts_dim: int, 
        interaction_fts_dim: int, 
        obstacle_fts_dim: int, 
        hidden_dims: list[int],
        dropout: float
    ):
        super(FusionNet, self).__init__()
        self.individual_fts_dim = individual_fts_dim
        self.interaction_fts_dim = interaction_fts_dim
        self.obstacle_fts_dim = obstacle_fts_dim
        self.hidden_dims = hidden_dims
        self.output_dim = 2
        self.dropout = dropout

        self.fc_interaction = nn.Linear(interaction_fts_dim, hidden_dims[0])
        self.fc_obstacle = nn.Linear(obstacle_fts_dim, hidden_dims[0])

        self.fcs_individual = self._build_mlp(individual_fts_dim, hidden_dims[1:], dropout)
        self.fcs_interaction = self._build_mlp(hidden_dims[0], hidden_dims[1:], dropout)
        self.fcs_obstacle = self._build_mlp(hidden_dims[0], hidden_dims[1:], dropout)

        self.fc_out_individual = nn.Linear(hidden_dims[-1], self.output_dim)
        self.fc_out_interaction = nn.Linear(hidden_dims[-1], self.output_dim)
        self.fc_out_obstacle = nn.Linear(hidden_dims[-1], self.output_dim)

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

        individual_out = self.fc_out_individual(self.fcs_individual(x_individual))
        interaction_out = self.fc_out_interaction(self.fcs_interaction(x_interaction))
        obstacle_out = self.fc_out_obstacle(self.fcs_obstacle(x_obstacle))

        return individual_out + interaction_out + obstacle_out, (individual_out, interaction_out, obstacle_out)

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
    def from_weight_file(path: str, device: str | torch.device = "cpu") -> "FusionNet":
        checkpoint = torch.load(path, map_location=device)
        individual_fts_dim = checkpoint['individual_fts_dim']
        interaction_fts_dim = checkpoint['interaction_fts_dim']
        obstacle_fts_dim = checkpoint['obstacle_fts_dim']
        hidden_dims = checkpoint['hidden_dims']
        dropout = checkpoint['dropout']

        model = FusionNet(
            individual_fts_dim=individual_fts_dim,
            interaction_fts_dim=interaction_fts_dim,
            obstacle_fts_dim=obstacle_fts_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model
