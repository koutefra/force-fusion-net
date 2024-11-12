from abc import ABC, abstractmethod
from entities.raw_scenes_data import RawSceneData

class BaseLoader(ABC):
    def __init__(self, path: str):
        self.path = path

    @abstractmethod
    def load_scene_by_id(self, scene_id: int) -> RawSceneData:
        pass
    
    @abstractmethod
    def load_scenes_by_ids(self, scene_ids: set[int]) -> RawSceneData:
        pass

    @abstractmethod
    def load_all_scenes(self) -> RawSceneData:
        pass