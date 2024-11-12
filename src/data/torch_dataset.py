import torch
from data.scene_dataset import SceneDataset
from torch.nn.utils.rnn import pad_sequence

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, scene_dataset: SceneDataset, device: torch.device = torch.device('cpu')):
        self.scene_dataset = scene_dataset
        self.ids = scene_dataset.get_ids()
        self.device = device

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[str, int, int]:
        return self.ids[idx]

    def prepare_batch(
        self, 
        ids: list[tuple[str, int, int]]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        raw_data = self.scene_dataset.get_frame_features_from_ids(ids)
        processed_data = [self.process_datapoint(x) for x in raw_data]

        input, output = zip(*processed_data)
        x_individual, x_interaction = zip(*input)
        x_individual_stack = torch.stack(x_individual)
        x_interaction_stack = pad_sequence(x_interaction, batch_first=True)
        output_stack = torch.stack(output).float()
        return (x_individual_stack, x_interaction_stack), output_stack

    def process_datapoint(
        self,
        raw_datapoint: tuple[dict[str, float], list[dict[str, float]]]
    ) -> tuple[tuple[list[float], list[list[float]]], tuple[float, float]]:
        individual_feature_dict, interaction_list = raw_datapoint
        
        x_individual = [
            value for key, value in individual_feature_dict.items() 
            if key not in {"acceleration_x", "acceleration_y"}
        ]
        x_interaction = [
            [value for key, value in interaction_dict.items() if key not in {"acceleration_x", "acceleration_y"}]
            for interaction_dict in interaction_list
        ]
        y_acceleration = torch.tensor(
            [individual_feature_dict["acceleration_x"], individual_feature_dict["acceleration_y"]], 
            dtype=torch.float32, 
            device=self.device
        )

        x_individual_ts = torch.tensor(x_individual, dtype=torch.float32, device=self.device)
        x_interaction_ts = (
            torch.tensor(x_interaction, dtype=torch.float32, device=self.device) if x_interaction else torch.empty(0, device=self.device)
        )
        return (x_individual_ts, x_interaction_ts), y_acceleration