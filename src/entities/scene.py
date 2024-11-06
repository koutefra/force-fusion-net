from dataclasses import dataclass
from entities.vector2d import Point2D
from entities.frame import Frame

@dataclass(frozen=True)
class Scene:
    id: int
    focus_person_id: int 
    focus_person_goal: Point2D
    fps: float
    frames: list[Frame]
    tag: list[int]
    dataset: str

Scenes = dict[int, Scene]