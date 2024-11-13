from abc import ABC, abstractmethod
from entities.raw_data import RawDataCollection

class BaseLoader(ABC):
    def __init__(self, paths: list[str]):
        self.paths = paths

    @abstractmethod
    def load_scenes_by_ids(self, scene_ids: set[int]) -> RawDataCollection:
        pass

    @abstractmethod
    def load_all_scenes(self) -> RawDataCollection:
        pass