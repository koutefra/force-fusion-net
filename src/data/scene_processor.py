from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from core.scene import Scene

class SceneProcessor(ABC):
    @abstractmethod
    def __init__(cls, scenes: Dict[int, Scene]):
        pass

    @abstractmethod
    def get_features(self) -> List[Dict[str, float]]:
        pass

    @abstractmethod
    def get_features_with_label(self) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        pass

    @abstractmethod
    def get_feature_index(self, feature_name: str) -> int:
        pass

    @abstractmethod
    def get_feature_name(self, index: int) -> str:
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass

    @abstractmethod
    def n_features(self) -> int:
        pass

    def label_dim(self) -> int:
        pass