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
        x_individual, x_interaction, x_obstacle = zip(*input_features)
        x_individual_stack = torch.stack(x_individual).to(self.device)
        x_interaction_stack = pad_sequence(x_interaction, batch_first=True).to(self.device)
        x_obstacle_stack = pad_sequence(x_obstacle, batch_first=True).to(self.device)
        labels_stack = torch.stack(labels).to(self.device)
        return (x_individual_stack, x_interaction_stack, x_obstacle_stack), labels_stack