from dataclasses import dataclass
from typing import Optional
from entities.vector2d import Point2D, Velocity, Acceleration

@dataclass
class RawSceneData:
    id: int
    goal_positions: dict[int, Point2D]
    obstacles: list[list[Point2D]]
    start_frame_number: int
    end_frame_number: int
    fps: float
    tag: Optional[list[int]] = None

@dataclass
class RawPersonTrackData:
    frame_number: int
    person_id: int
    position: Point2D
    velocity: Optional[Velocity] = None
    acceleration: Optional[Acceleration] = None

RawSceneTrajectories = dict[int, dict[int, RawPersonTrackData]]  # person_id -> frame_number -> track

@dataclass
class RawDataCollection:
    def __init__(
        self,
        scenes: list[RawSceneData],
        dataset_name: str,
        tracks: Optional[list[RawPersonTrackData]] = None,
        trajectories: Optional[dict[int, RawSceneTrajectories]] = None,
    ):
        if not tracks and not trajectories:
            raise ValueError("Must provide either tracks or trajectories")
        self.scenes = scenes
        self.dataset_name = dataset_name
        self.tracks = tracks
        self.trajectories = trajectories