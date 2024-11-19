from entities.scene import Scenes
from data.loaders.base_loader import BaseLoader
from entities.features import  SceneFeatures
from collections import defaultdict
import json
import pickle
from tqdm import tqdm

class SceneDataset:
    scenes = dict[str, Scenes]

    def __init__(self, loaders: dict[str, BaseLoader], print_progress: bool = True):
        self.loaders = loaders
        self.scenes = self._load(loaders, print_progress)
        self.print_progress = print_progress

    @staticmethod
    def _load(loaders: dict[str, BaseLoader], print_progress: bool = True):
        scenes = {}
        for loader_name, loader in loaders.items():
            scenes[loader_name] = loader.load(print_progress)
        return scenes

    def get_features(self) -> dict[str, dict[str, SceneFeatures]]:
        features = defaultdict(lambda: dict())
        for loader_name, loader_scenes in self.scenes.items():
            for scene_id, scene in tqdm(
                loader_scenes.items(), 
                desc=f"Extracting scene features of dataset {loader_name}...", 
                disable=not self.print_progress):
                scene_features = SceneFeatures.get_scene_features(scene)
                features[loader_name][scene_id] = scene_features
        return features

    def get_scene_features(self, loader_name: str, scene_id: str) -> SceneFeatures:
        return SceneFeatures.get_scene_features(self.scenes[loader_name][scene_id])

    @staticmethod
    def save_features(
        features: dict[str, dict[str, SceneFeatures]], 
        filepath: str, 
        save_format: str = "pickle"
    ) -> None:
        if save_format == "json":
            # Convert features to a JSON-compatible format
            json_compatible_features = [
                {
                    "loader": loader_name,
                    "scene": scene_id,
                    "frame": frame_number,
                    "person": person_id,
                    "features": person_features.to_json()
                }
                for loader_name, loader_scenes in features.items()
                for scene_id, scene_features in loader_scenes.items()
                for frame_number, frame_features in scene_features.features.items()
                for person_id, person_features in frame_features.features.items()
            ]
            with open(f"{filepath}.json", "w") as f:
                json.dump(json_compatible_features, f)
        elif save_format == "pickle":
            with open(f"{filepath}.pkl", "wb") as f:
                pickle.dump(features, f)
        else:
            raise ValueError("Unsupported save format. Use 'json' or 'pickle'.")

    @staticmethod
    def load_features(filepath: str, save_format: str = "pickle") -> dict[str, dict[str, SceneFeatures]]:
        if save_format == "json":
            with open(f"{filepath}.json", "r") as f:
                json_features = json.load(f)
                features = {}
                for loader_name, loader_scenes in json_features.items():
                    features[loader_name] = {
                        scene_id: SceneFeatures.from_dict(scene_features) 
                        for scene_id, scene_features in loader_scenes.items()
                    }
                return features
        elif save_format == "pickle":
            with open(f"{filepath}.pkl", "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported save format. Use 'json' or 'pickle'.")