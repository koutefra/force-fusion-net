from typing import List, Tuple, TypedDict, Dict
from abc import ABC, abstractmethod
from pedestrian_dataset import PedestrianDataset

class Velocity(TypedDict):
    vx: float
    vy: float

class Force(TypedDict):
    fx: float
    fy: float

class Prediction(TypedDict):
    scene_id: int 
    preds: Dict[PedestrianDataset.PosId, Tuple[Velocity, Force]]

class PredictorBase(ABC):
    @abstractmethod
    def predict(self, scene: PedestrianDataset.Scene, **kwargs) -> Prediction:
        pass