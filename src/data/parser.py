from entities.raw_data import RawDataCollection, RawTrackData, RawSceneData, RawSceneTrajectories
from entities.scene import Scene, Scenes
from entities.vector2d import Velocity, Acceleration, Point2D
from entities.frame import Frame
from entities.frame_object import PersonInFrame
from collections import defaultdict
from tqdm import tqdm
from typing import Optional

class Parser:
    def __init__(
        self,
        goal_position_fill_method: Optional[str] = "last_position",
        nan_strategy: str = "zero",
        fdm_type: str = "backward",
        fdm_win_size: int = 2,
        print_progress: bool = True
    ):
        self.goal_position_fill_method = goal_position_fill_method
        self.nan_strategy = nan_strategy
        self.fdm_type = fdm_type
        self.fdm_win_size = fdm_win_size
        self.print_progress = print_progress

        if goal_position_fill_method and goal_position_fill_method != "last_position":
            raise ValueError(f"Only 'last_position' strategy is currently supported.")

        if nan_strategy != "zero":
            raise ValueError(f"Only zero NaN strategy is currently supported.")

        if fdm_type != "backward":
            raise ValueError(f"Only backward FDM is currently supported.")

    def convert_to_scenes(self, data: RawDataCollection) -> Scenes:
        if data.tracks:
            data_paired = self.assign_tracks_to_scenes(data, dataset_name=data.dataset_name)
            if data.trajectories:
                raise ValueError("Expected data.trajectories to be None")
            data.trajectories = self.compute_trajectories(data_paired)

        scenes = {}
        for raw_scene in tqdm(
            data.scenes, 
            desc=f'[dataset_name={data.dataset_name}] Converting raw data into scenes...',
            disable=not self.print_progress):
            scene_trajectories = data.trajectories[raw_scene.id]
            scene = self.process_scene(raw_scene, scene_trajectories, data.dataset_name)
            scenes[scene.id] = scene
        return scenes

    def process_scene(
        self,
        raw_scene: RawSceneData, 
        scene_trajectories: RawSceneTrajectories,
        dataset_name: str
    ) -> Scene:
        scene_trajectories = self.compute_vel_and_acc_backward_fdm(
            scene_trajectories,
            fps=raw_scene.fps,
            dataset_name=dataset_name
        )
        scene_frames = self.trajectories_to_frames(scene_trajectories)

        if self.goal_position_fill_method == "last_position":
            raw_scene.goal_positions = self.extract_scene_goal_positions(scene_trajectories)

        return Scene(
            id=raw_scene.id,
            focus_person_ids=raw_scene.focus_person_ids,
            goal_positions=raw_scene.goal_positions,
            fps=raw_scene.fps,
            frames=scene_frames,
            tag=[] if raw_scene.tag is None else raw_scene.tag,
            dataset=dataset_name
        )

    def get_frame_counts(self, data: RawDataCollection) -> dict[int, int]:
        scene_lengths = {}
        if data.tracks:
            data_paired = self.assign_tracks_to_scenes(data, dataset_name=data.dataset_name)
            if data.trajectories:
                raise ValueError("Expected data.trajectories to be None")

            for scene_id, frames in data_paired.items():
                scene_lengths[scene_id] = len(frames)

        elif data.trajectories:
            for scene_id, persons in data.trajectories.items():
                scene_frame_numbers = set()
                for frames in persons.values():
                    scene_frame_numbers.update(frames.keys())
                scene_lengths[scene_id] = len(scene_frame_numbers)
        return scene_lengths

    @staticmethod
    def trajectories_to_frames(scene_trajectories: RawSceneTrajectories) -> list[Frame]:
        frame_person_dict = defaultdict(lambda: defaultdict(list))
        for person_id, frames in scene_trajectories.items():
            for frame_number in frames.keys():
                frame_person_dict[frame_number].append(person_id)

        frames = []
        for frame_number, person_ids in frame_person_dict.items():
            frame_objects = []
            for person_id in person_ids:
                track = scene_trajectories[person_id][frame_number]
                frame_object = PersonInFrame(
                    id=track.object_id,
                    position=track.position,
                    velocity=track.velocity,
                    acceleration=track.acceleration
                )
                frame_objects.append(frame_object)
            frame = Frame(number=frame_number, frame_objects=frame_objects)
            frames.append(frame)
        return frames

    @staticmethod
    def extract_scene_goal_positions(scene_trajectories: RawSceneTrajectories) -> dict[int, Point2D]:
        goal_positions = {}
        for person_id, frames in scene_trajectories.items():
            latest_frame_number = max(frames.keys())
            latest_track = frames[latest_frame_number]
            goal_positions[person_id] = latest_track.position
        return goal_positions

    def compute_vel_and_acc_backward_fdm(
        self,
        trajectories: RawSceneTrajectories,
        fps: float,
        dataset_name: str
    ) -> RawSceneTrajectories:
        def compute_fdm_property(person_trajectory: dict[int, RawTrackData], in_prop: str, out_prop: str):
            frame_numbers = list(person_trajectory.keys())
            for frame_id, frame_number in enumerate(person_trajectory.keys()):
                window_frame_ids = list(range(max(0, frame_id - self.fdm_win_size + 1), frame_id + 1))
                window = [
                    getattr(person_trajectory[fid], in_prop) 
                    for fid in window_frame_ids
                    if frame_numbers[fid] in person_trajectory
                ]
                if len(window) < 2:
                    computed_value = Velocity.zero() if out_prop == "velocity" else Acceleration.zero()
                else:
                    time_window = [
                        (frame_numbers[j] - frame_numbers[i]) / fps
                        for i, j in zip([id - 1 for id in window_frame_ids][1:], window_frame_ids[1:])
                    ]
                    computed_value = Velocity.from_points(window, time_window) if out_prop == "velocity" \
                        else Acceleration.from_velocities(window, time_window)
                setattr(person_trajectory[frame_number], out_prop, computed_value)

        # Compute velocities and accelerations
        for person_trajectory in tqdm(
            trajectories.person_trajectories.values(), 
            desc=f"[dataset_name={dataset_name}] Computing velocities and accelerations...", 
            disable=not self.print_progress):
            person_trajectory.frames = dict(sorted(person_trajectory.frames.items()))
            compute_fdm_property(person_trajectory.frames, "position", "velocity")
            compute_fdm_property(person_trajectory.frames, "velocity", "acceleration")

        return trajectories

    def assign_tracks_to_scenes(
        self,
        data: RawDataCollection,
        dataset_name: str = None
    ) -> dict[int, dict[int, dict[int, RawTrackData]]]:
        tracks_for_scenes = defaultdict(lambda: defaultdict(dict))
        
        # Create a lookup table to map frame numbers to relevant scenes
        frame_to_scene_ids = defaultdict(list)
        for scene in data.scenes:
            for frame in range(scene.start_frame_number, scene.end_frame_number + 1):
                frame_to_scene_ids[frame].append(scene.id)

        desc = f"[dataset_name={dataset_name}] Associating tracks with scenes..." \
            if dataset_name is not None else "Associating tracks with scenes..."
        for track in tqdm(data.tracks, desc=desc, disable=not self.print_progress):
            relevant_scene_ids = frame_to_scene_ids.get(track.frame_number, [])
            for scene_id in relevant_scene_ids:
                tracks_for_scenes[scene_id][track.frame_number].setdefault(track.object_id, track)

        return tracks_for_scenes

    @staticmethod
    def compute_trajectories(
        data: dict[dict[int, dict[int, RawTrackData]]]
    ) -> dict[int, RawSceneTrajectories]:
        trajectories = {}
        for scene_id, scene_data in data.items():
            trajectories[scene_id] = Parser.compute_scene_trajectories(scene_data)
        return trajectories

    @staticmethod
    def compute_scene_trajectories(
        scene_data: dict[int, dict[int, RawTrackData]]
    ) -> RawSceneTrajectories:
        temp_trajectories = defaultdict(lambda: defaultdict(dict))
        for frame_number, tracks in scene_data.items():
            for person_id, track in tracks.items():
                temp_trajectories[person_id][frame_number] = track

        trajectories = {
            obj_id: frames
            for obj_id, frames in temp_trajectories.items()
        }
        
        return RawSceneTrajectories(trajectories=trajectories)