from abc import ABC, abstractmethod
from typing import List
from core.scene import Scene

class BaseLoader(ABC):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    @abstractmethod
    def load(self, path: str) -> None:
        """Load data from a file path."""
        pass

    @abstractmethod
    def preprocess(self) -> List[Scene]:
        """Preprocess loaded data into scenes and tracks."""
        pass