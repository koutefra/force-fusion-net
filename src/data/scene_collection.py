from data.base_loader import BaseLoader 
from sklearn.model_selection import train_test_split
from entities.scene import Scene
from data.raw_scenes_processor import RawScenesProcessor
from data.feature_extractor import FeatureExtractor

class SceneCollection:
    scenes: dict[int, Scene]

    def __init__(self, loaders: list[BaseLoader]):
        self.scenes = {}
        for loader in loaders:
            raw_scenes = loader.load_scenes()
            scenes = RawScenesProcessor.process_raw_scenes(raw_scenes, loader.path, print_progress=True)
            self.scenes = {**self.scenes, **scenes}

    @classmethod
    def from_scenes(cls, scenes: dict[int, Scene]) -> "SceneCollection":
        instance = cls.__new__(cls)  # Bypass __init__ to manually set attributes
        instance.scenes = scenes
        return instance

    def get_scenes_as_features(self) -> dict[int, list[tuple[dict[str, float], list[dict[str, float]]]]]:
        return FeatureExtractor.extract_all_scenes_features(self.scenes)

    def split(self, test_ratio: float = 0.2, seed: int = 21) -> tuple["SceneCollection", "SceneCollection"]:
        """Split the dataset into train and test sets, preserving scene IDs."""
        scene_list = list(self.scenes.values())
        train_scenes, val_scenes = train_test_split(scene_list, test_size=test_ratio, random_state=seed)
        
        train_scenes_dict = {scene.id: scene for scene in train_scenes}
        val_scenes_dict = {scene.id: scene for scene in val_scenes}
        
        train_dataset = SceneCollection.from_scenes(train_scenes_dict)
        val_dataset = SceneCollection.from_scenes(val_scenes_dict)
        
        return train_dataset, val_dataset