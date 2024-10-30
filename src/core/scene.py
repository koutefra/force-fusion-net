from dataclasses import dataclass
from typing import Dict, List, Tuple
from core.vector2d import Point2D, Velocity, Acceleration

@dataclass(frozen=True)
class Scene:
    id: int
    focus_person_ids: List[int] 
    fps: float
    trajectories: Dict[int, Dict[int, Point2D]]  # {frame_id: {person_id: Position}}
    obstacles: Dict[int, List[Point2D]]  # {frame_id: [Position1, Position2, ...]}
    tag: List[int]
    dataset: str

    @classmethod
    def from_raw_data(cls, raw_data: dict) -> "Scene":
        trajectories = {}
        for frame, person_id, pos_data in raw_data["trajectories"]:
            if frame not in trajectories:
                trajectories[frame] = {}
            trajectories[frame][person_id] = Point2D(x=pos_data["x"], y=pos_data["y"])

        return cls(
            id=raw_data["id"],
            focus_person_ids=raw_data["focus_person_ids"],
            fps=raw_data["fps"],
            trajectories=trajectories,
            obstacles=[Point2D(x=obs["x"], y=obs["y"]) for obs in raw_data["obstacles"]],
            tag=raw_data["tag"],
            dataset=raw_data["dataset"]
        )

    @property
    def min_max_points(self) -> Tuple[Point2D, Point2D]:
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
        
        for _, persons in self.trajectories.items():
            for _, point in persons.items():
                min_x, min_y = min(min_x, point.x), min(min_y, point.y)
                max_x, max_y = max(max_x, point.x), max(max_y, point.y)
        
        return Point2D(x=min_x, y=min_y), Point2D(x=max_x, y=max_y)

    @property
    def sorted_frame_ids(self) -> List[int]:
        return sorted(set(frame_id for frame_id in self.trajectories.keys()))

    @property
    def sorted_person_ids(self) -> List[int]:
        return sorted(
            set(person_id for persons in self.trajectories.values() for person_id in persons.keys())
        )

    @property
    def person_frame_ranges(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        start_frames = {}
        end_frames = {}

        for person_id in self.sorted_person_ids:
            frames_for_person = [frame for frame, persons in self.trajectories.items() if person_id in persons]
            start_frames[person_id] = min(frames_for_person)
            end_frames[person_id] = max(frames_for_person)

        return start_frames, end_frames

    @property
    def velocities_central_difference(self) -> Dict[int, Dict[int, Velocity]]:
        velocities = {}
        frame_ids = self.sorted_frame_ids
        for level in range(1, len(frame_ids) - 1): 
            frame_id = frame_ids[level]
            prev_frame_id = frame_ids[level - 1]
            next_frame_id = frame_ids[level + 1]
            delta_time = (next_frame_id - prev_frame_id) / self.fps
            velocities[frame_id] = {}
            for person_id, pos in self.trajectories[frame_id].items():
                prev_pos = self.trajectories[prev_frame_id][person_id]
                next_pos = self.trajectories[next_frame_id][person_id]
                velocity = Velocity.from_points(prev_pos, next_pos, delta_time)
                velocities[frame_id][person_id] = velocity
        return velocities

    @property
    def accelerations_central_difference(self) -> Dict[int, Dict[int, Point2D]]:
        velocities = self.velocities_central_difference
        accelerations = {}
        frame_ids = list(velocities.keys())
        for level in range(2, len(frame_ids) - 2): 
            frame_id = frame_ids[level]
            prev_frame_id = frame_ids[level - 1]
            next_frame_id = frame_ids[level + 1]
            delta_time = (next_frame_id - prev_frame_id) / self.fps
            accelerations[frame_id] = {}
            for person_id, vel in velocities[frame_id].items():
                prev_vel = velocities[prev_frame_id][person_id]
                next_vel = velocities[next_frame_id][person_id]
                acceleration = Acceleration.from_velocities(prev_vel, next_vel, delta_time)
                accelerations[frame_id][person_id] = acceleration
        return accelerations