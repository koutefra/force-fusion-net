from abc import ABC, abstractmethod
from entities.raw_scenes import RawScenes

class BaseLoader(ABC):
    def __init__(self, path: str):
        self.path = path
    
    @abstractmethod
    def load_scenes(self) -> RawScenes:
        pass