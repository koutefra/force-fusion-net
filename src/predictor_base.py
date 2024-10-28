from typing import List, Tuple, TypedDict, Dict
from abc import ABC, abstractmethod
from pedestrian_dataset import PedestrianDataset
from vector2d import Velocity, Force

class Prediction(TypedDict):
    scene_id: int 
    preds: Dict[PedestrianDataset.PositionId, Tuple[Velocity, Force]]

class PredictorBase(ABC):
    @abstractmethod
    def predict(self, scene: PedestrianDataset.Scene, **kwargs) -> Prediction:
        pass