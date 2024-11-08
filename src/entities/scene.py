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

    @property
    def bounding_box(self) -> tuple[Point2D, Point2D]:
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        
        for frame in self.frames:
            for frame_obj in frame.frame_objects:
                pos = frame_obj.position
                min_x, min_y = min(min_x, pos.x), min(min_y, pos.y)
                max_x, max_y = max(max_x, pos.x), max(max_y, pos.y)
        
        return Point2D(x=min_x, y=min_y), Point2D(x=max_x, y=max_y)

Scenes = dict[int, Scene]