from typing import List, Tuple, TypedDict, Dict
from abc import ABC, abstractmethod
from pedestrian_dataset import PedestrianDataset

class Prediction(TypedDict):
    velocity: Tuple[float, float]
    force: Tuple[float, float]

Predictions= Dict[Tuple[PedestrianDataset.FrameNumber, PedestrianDataset.PedestrianId], Prediction]

class PredictorBase(ABC):
    @abstractmethod
    def predict(self, scene: "PedestrianDataset.Scene", **kwargs) -> Predictions:
        pass