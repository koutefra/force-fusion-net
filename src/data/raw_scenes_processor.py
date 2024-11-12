from tqdm import tqdm
from entities.vector2d import Point2D, Velocity, Acceleration
from entities.frame_object import PersonInFrame, FrameObject
from entities.frame import Frame
from entities.scene import Scene
from entities.raw_scenes_data import RawSceneData
from collections import defaultdict

class RawScenesProcessor:

    @staticmethod
    def process_raw_scenes(
        raw_scenes: RawSceneData, 
        dataset_name: str, 
        print_progress: bool = True
    ) -> dict[int, Scene]:
        raw_data_paired = RawScenesProcessor.assign_tracks_to_scenes(raw_scenes, print_progress)
        scenes = {}

        for raw_scene in tqdm(raw_scenes.scenes, desc=f"Processing {dataset_name} scene data...", disable=not print_progress):
            scene_id = raw_scene.id
            frames = RawScenesProcessor.process_frames(raw_data_paired[scene_id], raw_scene.fps)
            focus_person_goal = RawScenesProcessor.get_person_goal_position(raw_scene.person_id, frames)
            tag = RawScenesProcessor.process_tags(raw_scene.tag)
            
            scenes[scene_id] = Scene(
                id=scene_id,
                focus_person_id=raw_scene.person_id,
                focus_person_goal=focus_person_goal,
                fps=raw_scene.fps,
                frames=frames,
                tag=tag,
                dataset=dataset_name
            )

        return scenes

    @staticmethod
    def get_scene_lengths(raw_scenes: RawSceneData, print_progress: bool = True) -> dict[int, int]:
        ids = {}
        raw_data_paired = RawScenesProcessor.assign_tracks_to_scenes(raw_scenes, print_progress)
        for raw_scene in tqdm(raw_scenes.scenes, desc="Processing scene lengths data...", disable=not print_progress):
            scene_id = raw_scene.id
            frames = RawScenesProcessor.process_frames(raw_data_paired[scene_id], raw_scene.fps)
            ids[scene_id] = len(frames)
        return ids

    @staticmethod
    def get_person_goal_position(person_id: int, frames: list[Frame]) -> Point2D:
        for frame in reversed(frames):
            for frame_object in frame.frame_objects:
                if isinstance(frame_object, PersonInFrame) and frame_object.id == person_id:
                    return frame_object.position
        raise ValueError(f"Invalid person_id {person_id}. Person not found in any frame.")

    @staticmethod
    def process_frames(
        tracks_data: dict[int, dict[int, Point2D]], 
        fps: float,
        nan_strategy: str = "zero",
        backward_fdm_win_size: int = 2,
        print_progress: bool = True
    ) -> list[Frame]:
        if nan_strategy != "zero":
            raise ValueError(f"Only zero NaN strategy is currently supported.")
        frame_numbers = sorted(tracks_data.keys())
        trajectories = defaultdict(lambda: defaultdict(dict))

        # compute the trajectories data
        for frame_id in range(len(frame_numbers)):
            frame_number = frame_numbers[frame_id]
            frame_track_data = tracks_data[frame_number]
            for pid, pos in frame_track_data.items():
                trajectories[pid][frame_id] = {"pos": pos, "vel": Velocity.zero(), "acc": Acceleration.zero()} 

        # compute velocities and accelerations
        for prop in ["vel", "acc"]:
            for pid, frames in trajectories.items():
                frame_ids = list(frames.keys())
                for frame_id in frame_ids:
                    start_frame_id = max(0, frame_id - backward_fdm_win_size)
                    required_frame_ids = range(frame_id, frame_id - backward_fdm_win_size)
                    value_window = [
                        frames[fid][prop] for fid in required_frame_ids
                        if fid in frames
                    ]
                    frame_numbers_window = [frame_numbers[fid] for fid in required_frame_ids]
                        delta_times = [
                            (f_num_j - f_num_i) / fps for f_num_i, f_num_j in 
                            zip(frame_numbers_window, frame_numbers_window[1:])
                        ]
                    else:

                


                if i <= backward_fdm_win_size and nan_strategy == "last_value":
                    previous_frames = frames[frame_id - backward_fdm_win_size:frame_id]
                
            else:
                

            frame = Frame(number=frame_numbers[frame_id], frame_objects=[])
            frames.append(RawScenesProcessor.process_single_frame(i, frame_numbers, tracks_data, fps))

        return frames

    @staticmethod
    def process_single_frame(index: int, frame_numbers: list[int], tracks_data: dict[int, dict[int, Point2D]], fps: float) -> Frame:
        frame_number = frame_numbers[index]
        prev_prev_frame_number = frame_numbers[index - 2]
        prev_frame_number = frame_numbers[index - 1]
        next_frame_number = frame_numbers[index + 1]
        next_next_frame_number = frame_numbers[index + 2]

        persons_in_frame = []
        common_person_ids = set(tracks_data[prev_prev_frame_number].keys()) & set(tracks_data[prev_frame_number].keys()) & \
                            set(tracks_data[frame_number].keys()) & set(tracks_data[next_frame_number].keys()) & \
                            set(tracks_data[next_next_frame_number].keys())

        for person_id in common_person_ids:
            person_in_frame = RawScenesProcessor.compute_person_in_frame(
                person_id, 
                tracks_data, 
                [
                    prev_prev_frame_number, 
                    prev_frame_number, 
                    frame_number, 
                    next_frame_number, 
                    next_next_frame_number
                ], 
                fps
            )
            persons_in_frame.append(person_in_frame)

        return Frame(frame_number, persons_in_frame)

    @staticmethod
    def compute_person_in_frame(person_id: int, tracks_data: dict[int, dict[int, Point2D]], frame_numbers: list[int], fps: float) -> PersonInFrame:
        points = [(tracks_data[frame_number][person_id], frame_number) for frame_number in frame_numbers]
        position, velocity, acceleration = RawScenesProcessor.compute_motion_parameters(*points, fps=fps)
        return PersonInFrame(id=person_id, position=position, velocity=velocity, acceleration=acceleration)

    @staticmethod
    def backward_fdm(
        positions: list[Point2D],
        delta_times: list[int]
    ) -> tuple[Point2D, Velocity, Acceleration]:
        pos, frame_number = points[0]
        vel = Velocity.from_points(positions, delta_times)
    
        # old function
        prev_prev_point, prev_prev_frame = prev_prev_frame_point
        prev_point, prev_frame = prev_frame_point
        cur_point, cur_frame = cur_frame_point
        next_point, next_frame = next_frame_point
        next_next_point, next_next_frame = next_next_frame_point

        delta_time_pp_to_prev = (prev_frame - prev_prev_frame) / fps
        delta_time_prev_to_cur = (cur_frame - prev_frame) / fps
        delta_time_cur_to_next = (next_frame - cur_frame) / fps
        delta_time_next_to_nn = (next_next_frame - next_frame) / fps

        cur_velocity = Velocity.from_points(
            prev_point, next_point, delta_time_prev_to_cur + delta_time_cur_to_next)

        prev_velocity = Velocity.from_points(
            prev_prev_point, cur_point, delta_time_pp_to_prev + delta_time_prev_to_cur)
        next_velocity = Velocity.from_points(
            cur_point, next_next_point, delta_time_cur_to_next + delta_time_next_to_nn)

        cur_acceleration = Acceleration.from_velocities(
            prev_velocity, next_velocity, delta_time_prev_to_cur + delta_time_cur_to_next)

        return cur_point, cur_velocity, cur_acceleration

    @staticmethod
    def process_tags(tag: int | list[int] | None) -> list[int]:
        return [] if tag is None else tag if isinstance(tag, list) else [tag]

    @staticmethod
    def assign_tracks_to_scenes(raw_scenes: RawSceneData, print_progress: bool) -> dict[int, dict[int, dict[int, Point2D]]]:
        tracks_for_scenes = defaultdict(lambda: defaultdict(dict))
        
        # Create a lookup table to map frame numbers to relevant scenes
        frame_to_scene_ids = defaultdict(list)
        for scene in raw_scenes.scenes:
            for frame in range(scene.start_frame_number, scene.end_frame_number + 1):
                frame_to_scene_ids[frame].append(scene.id)

        for track in tqdm(raw_scenes.tracks, 
                          desc='Associating tracks with scenes...', 
                          disable=not print_progress):
            relevant_scene_ids = frame_to_scene_ids.get(track.frame_number, [])
            for scene_id in relevant_scene_ids:
                tracks_for_scenes[scene_id][track.frame_number].setdefault(track.person_id, Point2D(x=track.x, y=track.y))

        return tracks_for_scenes