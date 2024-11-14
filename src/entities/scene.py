from dataclasses import dataclass
from entities.vector2d import Point2D, Acceleration
from entities.frame import Frame, PersonInFrame
from typing import Callable

@dataclass(frozen=True)
class Obstacle:
    id: int
    vertices: list[Point2D] 

@dataclass(frozen=True)
class PersonInScene:
    id: int
    goal: Point2D 

@dataclass(frozen=True)
class Scene:
    id: int
    obstacles: list[Obstacle]
    persons: list[PersonInScene]
    frames: list[Frame]
    fps: float
    tag: list[int]
    dataset: str

    def calculate_delta_time(self, frame_cur_id: int):
        if frame_cur_id < 1 or frame_cur_id >= len(self.frames):
            raise ValueError(f"Invalid frame index {frame_cur_id}")
        return (self.frames[frame_cur_id].number - self.frames[frame_cur_id - 1].number) / self.fps

    def simulate_trajectory(self, predict_force_func: Callable[[Frame, int, Point2D], Acceleration]) -> list[PersonInFrame]:
        trajectory = []
        person = next(
            o for o in self.frames[0].frame_objects if isinstance(o, PersonInFrame) and o.id == self.focus_person_id
        )
        for frame_cur_id in range(len(self.frames)):
            adjusted_frame = self.frames[frame_cur_id].transform_object(person)
            pred_acc = predict_force_func(adjusted_frame, self.focus_person_id, self.focus_person_goal)

            person = PersonInFrame(
                id=person.id,
                position=person.position,
                velocity=person.velocity,
                acceleration=pred_acc
            )
            trajectory.append(person)

            if frame_cur_id < len(self.frames) - 1:
                delta_time = self.calculate_delta_time(frame_cur_id + 1)
                person = PersonInFrame(
                    id=person.id,
                    position=person.position + person.velocity * delta_time,
                    velocity=person.velocity + person.acceleration * delta_time,
                    acceleration=Acceleration.zero()  # placeholder for next iteration
                )

        return trajectory

    @property
    def bounding_box(self) -> tuple[Point2D, Point2D]:
        x_min, y_min = float("inf"), float("inf")
        x_max, y_max = float("-inf"), float("-inf")
        
        for frame in self.frames:
            for frame_obj in frame.frame_objects:
                pos = frame_obj.position
                x_min, y_min = min(x_min, pos.x), min(y_min, pos.y)
                x_max, y_max = max(x_max, pos.x), max(y_max, pos.y)
        
        return Point2D(x=x_min, y=y_min), Point2D(x=x_max, y=y_max)

Scenes = dict[int, Scene]