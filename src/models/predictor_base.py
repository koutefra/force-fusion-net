from typing import TypedDict, Optional, Dict
from abc import ABC, abstractmethod
from data.pedestrian_dataset import PedestrianDataset
from core.vector2d import Velocity, Acceleration 

class Prediction(TypedDict):
    scene_id: int 
    predicted_forces: Optional[Dict[PedestrianDataset.PositionId, Acceleration]]
    predicted_velocities: Optional[Dict[PedestrianDataset.PositionId, Velocity]]

class PredictorBase(ABC):
    @abstractmethod
    def predict(self, scene: PedestrianDataset.Scene, **kwargs) -> Prediction:
        pass