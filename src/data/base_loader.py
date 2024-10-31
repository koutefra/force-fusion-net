from abc import ABC, abstractmethod
from typing import Dict
from core.scene import Scene

class BaseLoader(ABC):
    @abstractmethod
    def load_scenes(self, path: str, dataset_name: str) -> Dict[int, Scene]:
        pass