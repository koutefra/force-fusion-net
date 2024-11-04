from typing import List, Tuple, Any, Dict
from core.scene import Scene
from data.base_loader import BaseLoader 
from sklearn.model_selection import train_test_split

class PedestrianDataset:
    scenes: Dict[int, Scene]

    def __init__(self, data_loader: BaseLoader, path: str, dataset_name: str):
        self._scenes = data_loader.load_scenes(path, dataset_name)

    @classmethod
    def from_scenes(cls, scenes: Dict[int, Scene]) -> "PedestrianDataset":
        instance = cls.__new__(cls)  # Bypass __init__ to manually set attributes
        instance._scenes = scenes
        return instance

    def get_scene_ids(self) -> List[int]:
        return list(self.scenes.keys())
        
    def get_scene_by_id(self, scene_id: int) -> Scene:
        if scene_id not in self.scenes:
            raise KeyError(f"Scene ID {scene_id} not found.")
        return self.scenes[scene_id]

    def get_scenes(self) -> Dict[int, Scene]:
        return self._scenes

    def split(self, test_ratio: float = 0.2, seed: int = 21) -> Tuple["PedestrianDataset", "PedestrianDataset"]:
        """Split the dataset into train and test sets, preserving scene IDs."""
        scene_list = list(self._scenes.values())
        train_scenes, val_scenes = train_test_split(scene_list, test_size=test_ratio, random_state=seed)
        
        train_scenes_dict = {scene.id: scene for scene in train_scenes}
        val_scenes_dict = {scene.id: scene for scene in val_scenes}
        
        train_dataset = PedestrianDataset.from_scenes(train_scenes_dict)
        val_dataset = PedestrianDataset.from_scenes(val_scenes_dict)
        
        return train_dataset, val_dataset