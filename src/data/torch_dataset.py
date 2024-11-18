import torch
from data.scene_dataset import SceneDataset
from data.feature_extractor import FeatureExtractor
from torch.nn.utils.rnn import pad_sequence
import random
from typing import Optional

class TorchSceneDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        scene_dataset: SceneDataset, 
        feature_extractor: FeatureExtractor,
        device: torch.device = torch.device('cpu')
    ):
        self.scene_dataset = scene_dataset
        self.feature_extractor = feature_extractor
        self.scene_ids = scene_dataset.get_all_scene_ids()
        self.scene_ids_list = list(self.scene_ids.items())
        self.device = device

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(
        self, 
        idx: int
    ) -> Optional[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        loader_name, scene_ids = self.scene_ids_list[idx]
        if not scene_ids:
            return None
        return self._get_features_and_label_for_id(loader_name, random.choice(scene_ids))

    def prepare_batch(
        self, 
        data: list[tuple[str, int]]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        data = [item for item in data if item is not None]

        if not data:
            raise ValueError("No valid data found for the provided IDs.")

        input_features, labels = zip(*data)
        x_individual, x_interaction, x_obstacle = zip(*input_features)

        x_individual_stack = torch.stack(x_individual).to(self.device)
        x_interaction_stack = pad_sequence(x_interaction, batch_first=True).to(self.device)
        x_obstacle_stack = pad_sequence(x_obstacle, batch_first=True).to(self.device)
        labels_stack = torch.stack(labels).float().to(self.device)
        return (x_individual_stack, x_interaction_stack, x_obstacle_stack), labels_stack

    def _get_features_and_label_for_id(
        self,
        loader_name: str,
        scene_id: int
    ) -> Optional[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
        scene = self.scene_dataset.get_scene(loader_name, scene_id)
        if not scene:
            return None

        frame_numbers = list(scene.frames.keys())
        frame_id = random.randint(0, len(frame_numbers) - 2)

        persons_in_frame = scene.frames[frame_numbers[frame_id]].persons
        persons_in_next_frame = scene.frames[frame_numbers[frame_id + 1]].persons
        common_person_ids = set(persons_in_frame.keys()) & set(persons_in_next_frame.keys())

        if not common_person_ids:
            return None

        selected_person_id = random.choice(list(common_person_ids))
        person_cur_frame = persons_in_frame[selected_person_id]
        person_next_frame = persons_in_next_frame[selected_person_id]
        features = self.feature_extractor.extract_person_in_frame_features(
            person_id=selected_person_id,
            frame=scene.frames[frame_numbers[frame_id]],
            obstacles=scene.obstacles,
            goal=scene.persons[selected_person_id].goal
        )
        position_change = person_next_frame.position - person_cur_frame.position
        individual_tensor, interaction_tensor, obstacle_tensor = features.to_tensor()
        label = torch.tensor([
            position_change.x,
            position_change.y
        ], dtype=torch.float32).to(self.device)
        return (individual_tensor, interaction_tensor, obstacle_tensor), label