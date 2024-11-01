from typing import Dict, TypedDict

class SceneDatapoint(TypedDict):
    person_features: Dict[str, float]
    interaction_features: Dict[str, float]
    obstacle_features: Dict[str, float]
    label: Dict[str, float]

SceneDatapoints = Dict[int, Dict[int, SceneDatapoint]]  # {frame_id: {person_id: SceneDatapoint}}