from typing import Optional
from entities.vector2d import Point2D, Velocity, Acceleration

class RawSceneData:
    id: int
    focus_person_ids: list[int]
    goal_positions: dict[int, Point2D]
    start_frame_number: int
    end_frame_number: int
    fps: float
    tag: Optional[list[int]]

class RawTrackData:
    frame_number: int
    object_id: int
    type: str
    position: Point2D
    velocity: Optional[Velocity]
    acceleration: Optional[Acceleration]

RawSceneTrajectories = dict[int, dict[int, RawTrackData]]  # person_id -> frame_number -> track

class RawDataCollection:
    def __init__(
        self,
        scenes: list[RawSceneData],
        dataset_name: str,
        tracks: Optional[list[RawTrackData]] = None,
        trajectories: Optional[dict[int, RawSceneTrajectories]] = None,
    ):
        if not tracks and not trajectories:
            raise ValueError("Must provide either tracks or trajectories")
        self.scenes = scenes
        self.dataset_name = dataset_name
        self.tracks = tracks
        self.trajectories = trajectories