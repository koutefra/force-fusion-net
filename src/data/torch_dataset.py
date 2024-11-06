import torch
from typing import Dict, Tuple, Callable, Any
from data.scene_collection import SceneCollection

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, scene_collection: SceneCollection):
        scene_collection.get_scenes_as_features()
        self._valid_scenes_indices_dict = scene_processor.get_all_valid_positions(scenes)
        self._sorted_scenes_indices_list = sorted(
            self._valid_scenes_indices_dict.keys(), key=lambda x: (x[0], x[1], x[2])
        )
        self._scenes = scenes
        self._scene_processor = scene_processor

    def __len__(self) -> int:
        return len(self._sorted_scenes_indices_list)

    def __getitem__(self, idx: int) -> Tuple[SceneDatapoint, Dict[str, int]]:
        scene_id, frame_id, person_id = self._sorted_scenes_indices_list[idx]
        scene = self._scenes[scene_id]
        datapoint = self._scene_processor.get_datapoint_from_position(scene, frame_id, person_id)
        return datapoint, {"scene_id": scene_id, "frame_id": frame_id, "person_id": person_id}

    def transform(self, transform):
        return TransformedTorchDataset(self, transform)

class TransformedTorchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
        self._dataset = dataset
        self._transform = transform

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        item = self._dataset[index]
        return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

    def transform(self, transform: Callable[..., Any]) -> "TransformedTorchDataset":
        return TransformedTorchDataset(self, transform)
