import torch
from typing import Dict, Tuple, List, Any, Callable
from data.pedestrian_dataset import PedestrianDataset
from core.scene import Scene

class TorchPedestrianDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: PedestrianDataset):
        self.dataset = dataset
        self.index_map = self._build_index_map()

    def _build_index_map(self) -> List[Tuple[int, int, int]]:
        index_map = []
        for scene_id in self.dataset.get_scene_ids():
            scene_datapoints = self.dataset.scene_to_datapoints(scene_id)
            for frame_id, persons in scene_datapoints.items():
                for person_id in persons.keys():
                    index_map.append((scene_id, frame_id, person_id))
        return index_map

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        """
        Retrieve a single data point by converting the relevant scene, frame, and person
        data to tensors on demand.

        Returns:
        - person_features: Tensor of person-related features
        - interaction_features: Tensor of interaction-related features
        - obstacle_features: Tensor of obstacle-related features
        - label: Tensor of target values
        - metadata: A dictionary containing `scene_id`, `frame_id`, and `person_id` for reference.
        """
        scene_id, frame_id, person_id = self.index_map[idx]

        scene_datapoints = self.dataset.scene_to_datapoints(scene_id)
        datapoint = scene_datapoints[frame_id][person_id]

        # Convert data to tensors
        person_features = torch.tensor(list(datapoint["person_features"].values()), dtype=torch.float32)
        interaction_features = torch.tensor(list(datapoint["interaction_features"].values()), dtype=torch.float32)
        obstacle_features = torch.tensor(list(datapoint["obstacle_features"].values()), dtype=torch.float32)
        label = torch.tensor(list(datapoint["label"].values()), dtype=torch.float32)

        # Metadata for reference
        metadata = {"scene_id": scene_id, "frame_id": frame_id, "person_id": person_id}

        return person_features, interaction_features, obstacle_features, label, metadata