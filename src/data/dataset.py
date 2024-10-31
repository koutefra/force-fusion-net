from typing import List, Tuple, Any, Dict
from core.scene import Scene
from data.base_loader import BaseLoader 
from data.feature_extractor import SceneFeatureExtractor
from sklearn.model_selection import train_test_split

class Dataset:
    scenes: Dict[int, Scene]

    def __init__(self, data_loader: BaseLoader, feature_extractor: SceneFeatureExtractor, path: str, dataset_name: str):
        self.scenes = data_loader.load_scenes(path, dataset_name)
        self.feature_extractor = feature_extractor

    @classmethod
    def from_scenes(cls, scenes: Dict[int, Scene], feature_extractor: SceneFeatureExtractor) -> "Dataset":
        instance = cls.__new__(cls)  # Bypass __init__ to manually set attributes
        instance.scenes = scenes
        instance.feature_extractor = feature_extractor
        return instance

    def to_datapoints(self) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        return [self.feature_extractor.scene_to_datapoints(scene) for scene in self.scenes.values()]

    def scene_to_datapoints(self, scene_id: int) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        if scene_id not in self.scenes:
            raise KeyError(f"Scene ID {scene_id} not found in dataset.")
        return self.feature_extractor.scene_to_datapoints(self.scenes[scene_id])

    def get_scene_by_id(self, scene_id: int) -> Scene:
        if scene_id not in self.scenes:
            raise KeyError(f"Scene ID {scene_id} not found.")
        return self.scenes[scene_id]

    def split(self, test_ratio: float = 0.2, seed: int = 21) -> Tuple["Dataset", "Dataset"]:
        """Split the dataset into train and test sets, preserving scene IDs."""
        scene_list = list(self.scenes.values())
        train_scenes, val_scenes = train_test_split(scene_list, test_size=test_ratio, random_state=seed)
        
        train_scenes_dict = {scene.id: scene for scene in train_scenes}
        val_scenes_dict = {scene.id: scene for scene in val_scenes}
        
        train_dataset = Dataset.from_scenes(train_scenes_dict, self.feature_extractor)
        val_dataset = Dataset.from_scenes(val_scenes_dict, self.feature_extractor)
        
        return train_dataset, val_dataset