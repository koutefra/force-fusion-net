from abc import ABC, abstractmethod
from entities.scene import Scenes

class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> Scenes:
        pass