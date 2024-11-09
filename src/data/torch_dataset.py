import torch
from data.scene_collection import SceneCollection
from torch.nn.utils.rnn import pad_sequence

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, scene_collection: SceneCollection):
        # dict[int, list[tuple[dict[str, float], list[dict[str, float]]]]]
        self.scenes_features = scene_collection.get_scenes_as_features()

        self.flattened_features = []
        self.index_to_scene_frame = {}

        for scene_id, features in self.scenes_features.items():
            for frame_id, feature in enumerate(features):
                self.flattened_features.append(feature)
                self.index_to_scene_frame[len(self.flattened_features) - 1] = (scene_id, frame_id)

    def __len__(self) -> int:
        return len(self.flattened_features)

    def __getitem__(self, idx: int) -> tuple[tuple[list[float], list[list[float]]], tuple[float, float]]:
        datapoint = self.flattened_features[idx]
        individual_feature_dict, interaction_list = datapoint

        x_individual = [
            value for key, value in individual_feature_dict.items() 
            if key not in {"acceleration_x", "acceleration_y"}
        ]
        x_interaction = [
            [value for key, value in interaction_dict.items() if key not in {"acceleration_x", "acceleration_y"}]
            for interaction_dict in interaction_list
        ]
        x_interaction = []
        y_acceleration = (
            individual_feature_dict["acceleration_x"],
            individual_feature_dict["acceleration_y"]
        )

        return self.prepare_example(x=(x_individual, x_interaction), y=y_acceleration)

    @staticmethod
    def prepare_example(
        x: tuple[list[float], list[list[float]]], 
        y: tuple[float, float]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        x_individual, x_interaction = x
        x_individual_ts = torch.tensor(x_individual, dtype=torch.float32)
        x_interaction_ts = (
            torch.tensor(x_interaction, dtype=torch.float32) if x_interaction else torch.empty(0)
        )
        y_ts = torch.tensor(y, dtype=torch.float32)
        return (x_individual_ts, x_interaction_ts), y_ts

    @staticmethod
    def prepare_batch(
        data: list[tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        input, output= zip(*data)
        x_individual, x_interaction = zip(*input)
        x_individual_stack = torch.stack(x_individual)
        x_interaction_stack = pad_sequence(x_interaction, batch_first=True)
        output_stack = torch.stack(output).float()
        return (x_individual_stack, x_interaction_stack), output_stack

    def get_scene_and_frame(self, idx: int) -> tuple[int, int]:
        return self.index_to_scene_frame[idx]