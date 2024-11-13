import random
from data.base_loader import BaseLoader 
from entities.scene import Scene, Scenes
from data.feature_extractor import FeatureExtractor
from data.processor import ProcessingStrategy, EagerProcessor, LazyProcessor
from collections import defaultdict

class SceneDataset:
    scene_lengths: dict[str, dict[int, int]]

    def __init__(self, loaders: dict[str, BaseLoader], load_on_demand: bool):
        self.processor = LazyProcessor(loaders) if load_on_demand else EagerProcessor(loaders)
        self.scene_lengths = self.processor.get_scene_lengths()

    @classmethod
    def from_processor(
        cls, 
        processor: ProcessingStrategy, 
        new_scene_lengths: dict[str, dict[int, int]]
    ) -> "SceneDataset":
        instance = cls.__new__(cls)
        instance.processor = processor
        instance.scene_lengths = new_scene_lengths
        return instance

    def get_scene(self, loader_name: str, id: int) -> Scene:
        return self.processor.get_scene(loader_name, id)

    def get_loader_scenes(self, loader_name: str) -> Scenes:
        return self.processor.get_loader_scenes(loader_name)

    def get_scenes(self, ids: dict[str, list[int]]) -> dict[str, Scenes]:
        return self.processor.get_scenes(ids)
        
    def get_all_scenes(self) -> dict[str, Scenes]:
        return self.processor.get_all_scenes()

    def get_loader_scene_lengths(self, loader_name: str) -> dict[int, int]:
        return self.scene_lengths[loader_name]

    def get_scene_lengths(self) -> dict[str, dict[int, int]]:
        return self.scene_lengths

    def get_ids(self) -> list[tuple[str,int, int]]:
        ids = []
        lengths = self.get_scene_lengths()
        for loader_name, scene_lengths in lengths.items():
            for scene_id, scene_len in scene_lengths.items():
                for frame_id in range(scene_len):
                    ids.append((loader_name, scene_id, frame_id))
        return ids

    def get_total_number_of_frames(self) -> int:
        n_frames = 0
        lengths = self.get_scene_lengths()
        for scene_lengths in lengths.values():
            for scene_length in scene_lengths.values():
                n_frames += scene_length
        return n_frames

    def get_frame_features(
        self,
        loader_name: str,
        scene_id: int,
        frame_id: int
    ) -> list[tuple[dict[str, float], list[dict[str, float]]]]:
        scene = self.get_scene(loader_name, scene_id)
        frame = scene.frames[frame_id]
        return FeatureExtractor.extract_frame_features(
            scene.focus_person_id, 
            frame.frame_objects, 
            scene.focus_person_goal
        )

    def get_scene_features(
        self,
        loader_name: str,
        scene_id: int
    ) -> list[tuple[dict[str, float], list[dict[str, float]]]]:
        scene = self.get_scene(loader_name, scene_id)
        return FeatureExtractor.extract_scene_features(scene)

    def get_frame_features_from_ids(
        self,
        ids: list[tuple[str, int, int]],
    ) -> list[tuple[dict[str, float], list[dict[str, float]]]]:
        ids_wo_frames: dict[str, list[int]] = defaultdict(set)
        for loader_name, scene_id, _ in ids:
            ids_wo_frames[loader_name].add(scene_id)
        ids_wo_frames = {loader_name: list(scene_ids) for loader_name, scene_ids in ids_wo_frames.items()}

        frame_features = []
        batch_scenes = self.get_scenes(ids_wo_frames)
        for scenes in batch_scenes.values():
            for scene in scenes.values():
                for frame in scene.frames:
                    cur_frame_features = FeatureExtractor.extract_frame_features(
                        scene.focus_person_id,
                        frame.frame_objects,
                        scene.focus_person_goal
                    )
                    frame_features.append(cur_frame_features)
        return frame_features

    def get_all_loader_scene_features(
        self,
        loader_name: str
    ) -> dict[int, list[tuple[dict[str, float], list[dict[str, float]]]]]:
        loader_scenes = self.get_loader_scenes(loader_name)
        return FeatureExtractor.extract_all_scenes_features(loader_scenes)

    def get_all_scene_features(
        self,
    ) -> dict[int, list[tuple[dict[str, float], list[dict[str, float]]]]]:
        scenes = self.get_all_scenes()
        return FeatureExtractor.extract_all_scenes_features(scenes)

    def split(self, test_ratio: float = 0.2, seed: int = 21) -> tuple["SceneDataset", "SceneDataset"]:
        lengths = self.get_scene_lengths()
        all_scene_ids = [(loader_name, scene_id) for loader_name, scenes in lengths.items() for scene_id in scenes]

        random.shuffle(all_scene_ids)
        split_index = int(len(all_scene_ids) * (1 - test_ratio))
        train_ids = all_scene_ids[:split_index]
        test_ids = all_scene_ids[split_index:]

        # Organize into dictionaries compatible with `filter_scenes`
        def organize_ids(scene_ids: list[tuple[str, int]]) -> dict[str, list[int]]:
            organized = {}
            for loader_name, scene_id in scene_ids:
                organized.setdefault(loader_name, []).append(scene_id)
            return organized

        train_dict = organize_ids(train_ids)
        test_dict = organize_ids(test_ids)

        train_scene_lengths = {
            loader_name: {scene_id: lengths[loader_name][scene_id] for scene_id in train_dict.get(loader_name, [])}
            for loader_name in train_dict
        }
        test_scene_lengths = {
            loader_name: {scene_id: lengths[loader_name][scene_id] for scene_id in test_dict.get(loader_name, [])}
            for loader_name in test_dict
        }

        train_processor = self.processor.filter_scenes(test_dict)
        test_processor = self.processor.filter_scenes(train_dict)

        train_dataset = SceneDataset.from_processor(train_processor, train_scene_lengths) 
        test_dataset = SceneDataset.from_processor(test_processor, test_scene_lengths) 

        return train_dataset, test_dataset