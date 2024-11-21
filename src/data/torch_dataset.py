import torch
from entities.features import LabeledFeatures
from torch.nn.utils.rnn import pad_sequence

class TorchSceneDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data: list[LabeledFeatures],
        device: torch.device = torch.device('cpu'),
        dtype: torch.dtype = torch.float32
    ):
        self.data = data
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, 
        idx: int
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        datapoint = self.data[idx]
        x, y = datapoint.to_tensor(self.device, self.dtype)
        return x, y

    def prepare_batch(
        self, 
        data: list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        input_features, labels = zip(*data)

        # input
        x_individual, x_interaction, x_obstacle = zip(*input_features)
        x_individual_stack = torch.stack(x_individual).to(self.device)
        x_interaction_stack = pad_sequence(x_interaction, batch_first=True).to(self.device)
        x_obstacle_stack = pad_sequence(x_obstacle, batch_first=True).to(self.device)

        # label
        cur_pos, next_pos, cur_vel, delta_times = zip(*labels)
        cur_pos_stack = torch.stack(cur_pos).to(self.device)
        next_pos_stack = torch.stack(next_pos).to(self.device)
        cur_vel_stack = torch.stack(cur_vel).to(self.device)
        delta_time_stack = torch.stack(delta_times).to(self.device)

        features = (x_individual_stack, x_interaction_stack, x_obstacle_stack)
        labels = (cur_pos_stack, next_pos_stack, cur_vel_stack, delta_time_stack)
        return features, labels