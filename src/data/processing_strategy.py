from abc import ABC, abstractmethod
from data.base_loader import BaseLoader
from entities.scene import Scene, Scenes
from data.raw_scenes_processor import RawScenesProcessor

class ProcessingStrategy(ABC):
    def __init__(
        self, 
        loaders: dict[str, BaseLoader], 
        disabled_scene_ids: dict[str, list[int]] = {}
    ):
        self.loaders = loaders
        self.disabled_scene_ids = {key: set(value) for key, value in disabled_scene_ids.items()}

    @abstractmethod
    def get_scene(self, loader_name: str, scene_id: int) -> Scene:
        pass

    @abstractmethod
    def get_scenes(self, ids: dict[str, list[int]]) -> dict[str, Scenes]:
        pass

    @abstractmethod
    def filter_scenes(self, ids: dict[str, list[int]]) -> "ProcessingStrategy":
        pass

    @abstractmethod
    def get_loader_scenes(self, loader_name: str) -> Scenes:
        pass

    @abstractmethod
    def get_all_scenes(self) -> dict[str, Scenes]:
        pass
    
    @abstractmethod
    def get_loader_scene_lengths(self, loader_name: str) -> dict[int, int]:
        pass

    @abstractmethod
    def get_scene_lengths(self) -> dict[str, dict[int, int]]:
        pass

    def _fetch_loader_scenes(self, loader_name: str) -> Scenes:
        raw_scenes = self.loaders[loader_name].load_all_scenes()
        scenes = RawScenesProcessor.process_raw_scenes(raw_scenes, loader_name)
        return {
            scene_id: scene 
            for scene_id, scene in scenes.items()
            if scene_id not in self.disabled_scene_ids.get(loader_name, set())
        }

    def _fetch_all_scenes(self) -> dict[str, Scenes]:
        all_scenes = {}
        for loader_name in self.loaders.keys():
            scenes = self._fetch_loader_scenes(loader_name)
            if loader_name not in all_scenes:
                all_scenes[loader_name] = {}
            all_scenes[loader_name].update(scenes)
        return all_scenes

class LazyProcessor(ProcessingStrategy):
    def get_scene(self, loader_name: str, scene_id: int) -> Scene:
        if scene_id in self.disabled_scene_ids.get(loader_name, set()):
            raise ValueError(f"Key {(loader_name, scene_id)} disabled!")
        raw_scene = self.loaders[loader_name].load_scene_by_id(scene_id)
        return RawScenesProcessor.process_raw_scenes(raw_scene, loader_name).get(scene_id)

    def get_scenes(self, ids: dict[str, list[int]]) -> dict[str, Scenes]:
        all_scenes = {}
        for loader_name, scene_ids in ids.items():
            valid_scene_ids = {
                scene_id for scene_id in scene_ids 
                if (loader_name, scene_id) not in self.disabled_scene_ids
            }
            raw_scenes = self.loaders[loader_name].load_scenes_by_ids(valid_scene_ids)
            scenes = RawScenesProcessor.process_raw_scenes(raw_scenes, loader_name)
            all_scenes[loader_name] = scenes
        return all_scenes

    def filter_scenes(self, ids: dict[str, list[int]]) -> "LazyProcessor":
        disabled_scene_ids_copy = {key: value.copy() for key, value in self.disabled_scene_ids.items()}
        for key, new_ids in ids.items():
            if key in disabled_scene_ids_copy:
                disabled_scene_ids_copy[key].update(new_ids)
            else:
                disabled_scene_ids_copy[key] = set(new_ids)
        return LazyProcessor(self.loaders, disabled_scene_ids_copy)

    def get_loader_scenes(self, loader_name: str) -> Scenes:
        return self._fetch_loader_scenes(loader_name)

    def get_all_scenes(self) -> dict[str, Scenes]:
        return self._fetch_all_scenes()

    def get_loader_scene_lengths(self, loader_name: str) -> dict[int, int]:
        raw_scenes = self.loaders[loader_name].load_all_scenes()
        return {
            scene_id: scene_len
            for scene_id, scene_len in RawScenesProcessor.get_scene_lengths(raw_scenes).items()
            if scene_id not in self.disabled_scene_ids.get(loader_name, set())
        }

    def get_scene_lengths(self) -> dict[str, dict[int, int]]:
        lengths = {}
        for loader_name in self.loaders.keys():
            scene_lengths = self.get_loader_scene_lengths(loader_name)
            lengths[loader_name] = scene_lengths
        return lengths

class EagerProcessor(ProcessingStrategy):
    def __init__(
        self, 
        loaders: dict[str, BaseLoader] = None, 
        disabled_scene_ids: dict[str, set[int]] = None,
        scenes: dict[str, dict[int, Scene]] = None
    ):
        super().__init__(loaders or {}, disabled_scene_ids or {})
        self._cache = scenes if scenes is not None else self._fetch_all_scenes()

    def get_scene(self, loader_name: str, scene_id: int) -> Scene:
        if (loader_name, scene_id) in self.disabled_scene_ids:
            return None
        return self._cache.get(loader_name, {}).get(scene_id)

    def get_scenes(self, ids: dict[str, list[int]]) -> dict[str, Scenes]:
        all_scenes = {}
        for loader_name, scene_ids in ids.items():
            loader_cached_scenes = self._cache.get(loader_name, {})
            loader_disabled_scene_ids = self.disabled_scene_ids.get(loader_name, set())
            all_scenes[loader_name] = {}
            for scene_id in scene_ids:
                if scene_id in loader_disabled_scene_ids:
                    continue
                scene = loader_cached_scenes.get(scene_id)
                if scene:
                    all_scenes[loader_name][scene_id] = scene
        return all_scenes

    def filter_scenes(self, ids: dict[str, list[int]]) -> "EagerProcessor":
        ids = {loader_name: set(scene_ids) for loader_name, scene_ids in ids.items()}
        filtered_cache = {}
        for loader_name, scenes in self._cache.items():
            if loader_name not in ids:
                continue

            valid_scene_ids = ids[loader_name]
            disabled_scene_ids = self.disabled_scene_ids.get(loader_name, set())
            filtered_scenes = {
                scene_id: scene
                for scene_id, scene in scenes.items()
                if scene_id in valid_scene_ids and scene_id not in disabled_scene_ids
            }
            if filtered_scenes:
                filtered_cache[loader_name] = filtered_scenes
        return EagerProcessor(scenes=filtered_cache, disabled_scene_ids=self.disabled_scene_ids)

    def get_loader_scenes(self, loader_name: str) -> Scenes:
        return {
            scene_id: scene
            for scene_id, scene in self._cache.get(loader_name, {}).items()
            if scene_id not in self.disabled_scene_ids.get(loader_name, set())
        }

    def get_all_scenes(self) -> dict[str, Scenes]:
        all_scenes = {}
        for loader_name, scenes in self._cache.items():
            enabled_scenes = {
                scene_id: scene
                for scene_id, scene in scenes.items()
                if scene_id not in self.disabled_scene_ids.get(loader_name, set())
            }
            if enabled_scenes:
                all_scenes[loader_name] = enabled_scenes
        return all_scenes

    def get_loader_scene_lengths(self, loader_name: str) -> dict[int, int]:
        scenes = self._cache[loader_name]
        scene_lengths = {}
        for scene_id, scene in scenes.items():
            if scene_id not in self.disabled_scene_ids.get(loader_name, set()):
                scene_lengths[scene_id] = len(scene.frames)
        return scene_lengths

    def get_scene_lengths(self) -> dict[str, dict[int, int]]:
        scene_lengths = {}
        for loader_name in self._cache.keys():
            scene_lengths[loader_name] = self.get_loader_scene_lengths(loader_name)
        return scene_lengths