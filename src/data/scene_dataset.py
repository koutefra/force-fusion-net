from entities.scene import Scenes
from data.loaders.base_loader import BaseLoader
from entities.features import  SceneFeatures, Features
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

    def _process_scenes(
        self,
        operation: callable,
        desc_template: str,
    ) -> dict[str, dict[str, SceneFeatures]]:
        features = defaultdict(lambda: dict())
        for loader_name, loader_scenes in self.scenes.items():
            for scene_id, scene in tqdm(
                loader_scenes.items(),
                desc=desc_template.format(loader_name=loader_name),
                disable=not self.print_progress
            ):
                features[loader_name][scene_id] = operation(scene)
        return features

    def get_features(self) -> dict[str, dict[str, SceneFeatures]]:
        return self._process_scenes(
            operation=SceneFeatures.get_scene_features,
            desc_template="Extracting scene features of dataset {loader_name}..."
        )

    def get_labeled_features(self) -> dict[str, dict[str, SceneFeatures]]:
        return self._process_scenes(
            operation=SceneFeatures.get_scene_labeled_features,
            desc_template="Extracting labeled scene features of dataset {loader_name}..."
        )

    def get_scene_features(self, loader_name: str, scene_id: str) -> SceneFeatures:
        return SceneFeatures.get_scene_features(self.scenes[loader_name][scene_id])

    def get_scene_labeled_features(self, loader_name: str, scene_id: str) -> SceneFeatures:
        return SceneFeatures.get_scene_labeled_features(self.scenes[loader_name][scene_id])

    @staticmethod
    def save_features_as_ndjson(
        features: dict[str, dict[str, SceneFeatures]], 
        filepath: str,
        writing_mode: str = "w"
    ) -> None:
        with open(f"{filepath}.ndjson", writing_mode) as f:
            for loader_name, loader_scenes in features.items():
                for scene_id, scene_features in loader_scenes.items():
                    for features_dict in scene_features.to_ndjson():
                        line_to_dump = {"loader": loader_name, "scene": scene_id, **features_dict}
                        json.dump(line_to_dump, f)
                        f.write("\n")
    @staticmethod
    def load_features_from_ndjson(filepath: str) -> dict[str, dict[str, SceneFeatures]]:
        features = defaultdict(dict)
        grouped_data = defaultdict(lambda: defaultdict(list))

        with open(filepath, "r") as file:
            for line in file:
                json_object = json.loads(line)
                loader_name = json_object["loader"]
                scene_id = json_object["scene"]
                grouped_data[loader_name][scene_id].append({
                    "frame_number": json_object["frame_number"],
                    "person": json_object["person"],
                    "features": json_object["features"],
                })

        for loader_name, loader_scenes in grouped_data.items():
            for scene_id, scene_data in loader_scenes.items():
                features[loader_name][scene_id] = SceneFeatures.from_dict(scene_data)

        return features

    @staticmethod
    def save_features_as_pickle(
        features: dict[str, dict[str, SceneFeatures]], 
        filepath: str
    ) -> None:
        with open(f"{filepath}.pkl", "wb") as f:
            pickle.dump(features, f)

    @staticmethod
    def load_features_from_pickle(filepath: str) -> dict[str, dict[str, SceneFeatures]]:
        with open(filepath, "rb") as f:
            features = pickle.load(f)
        return features