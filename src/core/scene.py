from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from core.vector2d import Point2D, Velocity, Acceleration
from functools import cached_property

@dataclass(frozen=True)
class Scene:
    id: int
    focus_person_ids: List[int] 
    focus_person_goals: Dict[int, Point2D]  # {person_id: Goal Point}
    fps: float
    trajectories: Dict[int, Dict[int, Point2D]]  # {frame_id: {person_id: Point}}
    obstacles: Dict[int, List[Point2D]]  # {frame_id: [Point1, Point2, ...]}
    tag: List[int]
    dataset: str

    @classmethod
    def from_raw_data(cls, raw_data: dict) -> "Scene":
        return cls(
            id=raw_data["id"],
            focus_person_ids=raw_data["focus_person_ids"],
            focus_person_goals=raw_data["focus_person_goals"],
            fps=raw_data["fps"],
            trajectories=raw_data["trajectories"],
            obstacles=raw_data["obstacles"],
            tag=raw_data["tag"],
            dataset=raw_data["dataset"]
        )

    @cached_property
    def min_max_points(self) -> Tuple[Point2D, Point2D]:
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        
        for _, persons in self.trajectories.items():
            for _, point in persons.items():
                min_x, min_y = min(min_x, point.x), min(min_y, point.y)
                max_x, max_y = max(max_x, point.x), max(max_y, point.y)
        
        return Point2D(x=min_x, y=min_y), Point2D(x=max_x, y=max_y)

    @cached_property
    def sorted_frame_ids(self) -> List[int]:
        return sorted(set(frame_id for frame_id in self.trajectories.keys()))

    @property
    def frame_ids(self) -> List[int]:
        return set(frame_id for frame_id in self.trajectories.keys())
        
    @property
    def person_ids(self) -> List[int]:
        return set(person_id for persons in self.trajectories.values() for person_id in persons.keys())

    @property
    def persons_ids_in_frame(self, frame_id) -> List[int]:
        return list(self.trajectories.get(frame_id, {}).keys())

    @cached_property
    def person_frame_ranges(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        start_frames = {}
        end_frames = {}

        for person_id in self.sorted_person_ids:
            frames_for_person = [frame for frame, persons in self.trajectories.items() if person_id in persons]
            start_frames[person_id] = min(frames_for_person)
            end_frames[person_id] = max(frames_for_person)

        return start_frames, end_frames

    def _central_difference(self, attr: str, calc_method: Callable) -> Dict[int, Dict[int, Point2D]]:
        differences = {}
        data = getattr(self, attr)
        frame_ids = list(data.keys())
        for level in range(1, len(frame_ids) - 1):
            frame_id = frame_ids[level]
            prev_frame_id = frame_ids[level - 1]
            next_frame_id = frame_ids[level + 1]

            delta_time = (next_frame_id - prev_frame_id) / self.fps
            differences[frame_id] = {}

            person_ids = set(data[prev_frame_id].keys()) & \
                        set(data[frame_id].keys()) & \
                        set(data[next_frame_id].keys())

            for person_id in person_ids:
                prev = data[prev_frame_id][person_id]
                next = data[next_frame_id][person_id]
                difference = calc_method(prev, next, delta_time)
                differences[frame_id][person_id] = difference
        return differences

    @cached_property
    def velocities_central_difference(self) -> Dict[int, Dict[int, Velocity]]:
        return self._central_difference(
            attr="trajectories",
            calc_method=Velocity.from_points
        )

    @cached_property
    def accelerations_central_difference(self) -> Dict[int, Dict[int, Acceleration]]:
        return self._central_difference(
            attr="velocities_central_difference",
            calc_method=Acceleration.from_velocities
        )