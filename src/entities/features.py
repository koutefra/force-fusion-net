from dataclasses import dataclass, asdict
from collections import defaultdict
from entities.scene import Scene, Obstacle, Frame, Person
from entities.vector2d import Point2D, Velocity
import torch
import json
from typing import Any
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

@dataclass(frozen=True)
class FeatureBase:
    def to_list(self, precision: int = 6) -> list[float]:
        return [round(value, precision) if isinstance(value, float) else value for value in asdict(self).values()]

    @classmethod
    def dim(cls) -> int:
        return len(cls.__dataclass_fields__)

    def to_tensor(self, device: torch.device | str, dtype: torch.dtype = torch.float32, precision: int = 6) -> torch.Tensor:
        return torch.tensor(self.to_list(precision), dtype=dtype, device=device)

    def to_json(self, precision: int = 6) -> dict:
        return {key: (round(value, precision) if isinstance(value, float) else value) for key, value in asdict(self).items()}

    @classmethod
    def from_list(cls, values: list[float]) -> "FeatureBase":
        if len(values) != cls.dim():
            raise ValueError(f"Expected {cls.dim()} values, but got {len(values)}.")
        return cls(*values)

@dataclass(frozen=True)
class IndividualFeatures(FeatureBase):
    vel_x: float
    vel_y: float
    dist_to_goal: float
    dir_x_to_goal: float
    dir_y_to_goal: float
    vel_towards_goal: float

    @staticmethod
    def get_individual_features(person: Person, goal_pos: Point2D) -> "IndividualFeatures":
        dist_to_goal = (person.position - goal_pos).magnitude()
        dir_to_goal = person.position.direction_to(goal_pos)
        vel_towards_goal = person.velocity.dot(dir_to_goal)
        return IndividualFeatures(
            vel_x=person.velocity.x,
            vel_y=person.velocity.y,
            dist_to_goal=dist_to_goal,
            dir_x_to_goal=dir_to_goal.x,
            dir_y_to_goal=dir_to_goal.y,
            vel_towards_goal=vel_towards_goal
        )

@dataclass(frozen=True)
class InteractionFeatures(FeatureBase):
    dist_to_p: float
    dir_x_to_p: float
    dir_y_to_p: float
    rel_vel_to_p: float
    alignment_to_p: float

    @staticmethod
    def get_interaction_features(person: Person, person_id: int, frame: Frame) -> list["InteractionFeatures"]:
        interaction_features = []
        for other_person_id, other_person in frame.items():
            if person_id == other_person_id or other_person.velocity is None:
                continue
            
            distance = (person.position - other_person.position).magnitude()
            direction_vector = person.position.direction_to(other_person.position)
            relative_velocity = (person.velocity - other_person.velocity).magnitude()
            alignment = person.velocity.dot(direction_vector)
            
            interaction_features.append(InteractionFeatures(
                dist_to_p=distance,
                dir_x_to_p=direction_vector.x,
                dir_y_to_p=direction_vector.y,
                rel_vel_to_p=relative_velocity,
                alignment_to_p=alignment,
            ))
        return interaction_features

@dataclass(frozen=True)
class ObstacleFeatures(FeatureBase):
    dist_to_o_cls: float
    dir_x_to_o_cls: float
    dir_y_to_o_cls: float
    dist_to_o_start: float
    dir_x_to_o_start: float
    dir_y_to_o_start: float
    dist_to_o_end: float
    dir_x_to_o_end: float
    dir_y_to_o_end: float

    @staticmethod
    def get_obstacle_features(person: Person, obstacles: list[Obstacle]) -> list["ObstacleFeatures"]:
        obstacle_features = []
        for obstacle in obstacles:
            closest_point = closest_point_on_line(person.position, obstacle.start_point, obstacle.end_point)

            dist_to_closest = (person.position - closest_point).magnitude()
            dir_to_closest = person.position.direction_to(closest_point)

            dist_to_start = (person.position - obstacle.start_point).magnitude()
            dir_to_start = person.position.direction_to(obstacle.start_point)

            dist_to_end = (person.position - obstacle.end_point).magnitude()
            dir_to_end = person.position.direction_to(obstacle.end_point)

            obstacle_features.append(ObstacleFeatures(
                dist_to_o_cls=dist_to_closest,
                dir_x_to_o_cls=dir_to_closest.x,
                dir_y_to_o_cls=dir_to_closest.y,
                dist_to_o_start=dist_to_start,
                dir_x_to_o_start=dir_to_start.x,
                dir_y_to_o_start=dir_to_start.y,
                dist_to_o_end=dist_to_end,
                dir_x_to_o_end=dir_to_end.x,
                dir_y_to_o_end=dir_to_end.y
            ))
        return obstacle_features

@dataclass(frozen=True)
class Features:
    individual_features: IndividualFeatures
    interaction_features: list[InteractionFeatures]
    obstacle_features: list[ObstacleFeatures]

    @staticmethod
    def get_features(person: Person, person_id: int, frame: Frame, obstacles: list[Obstacle]) -> "Features":
        if person.velocity is None or person.goal is None:
            raise ValueError(f'Person {person_id} has either velocity, goal, or both equal to None')
        individual_features = IndividualFeatures.get_individual_features(person, person.goal)
        interaction_features = InteractionFeatures.get_interaction_features(person, person_id, frame)
        obstacle_features = ObstacleFeatures.get_obstacle_features(person, obstacles)
        return Features(
            individual_features=individual_features,
            interaction_features=interaction_features,
            obstacle_features=obstacle_features
        )

    def to_tensor(
        self, 
        device: torch.device | str, 
        dtype: torch.dtype = torch.float32, 
        precision: int = 6
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        individual_tensor = self.individual_features.to_tensor(device, dtype, precision)
        interaction_tensors = (
            torch.stack([feat.to_tensor(device, dtype, precision) for feat in self.interaction_features])
            if self.interaction_features else torch.empty((0, InteractionFeatures.dim()))
        )
        obstacle_tensors = (
            torch.stack([feat.to_tensor(device, dtype, precision) for feat in self.obstacle_features])
            if self.obstacle_features else torch.empty((0, ObstacleFeatures.dim()))
        )
        return individual_tensor, interaction_tensors, obstacle_tensors

    def to_ndjson(self) -> dict[str, Any]:
        return {
            "individual": self.individual_features.to_list(),
            "interaction": [feat.to_list() for feat in self.interaction_features],
            "obstacle": [feat.to_list() for feat in self.obstacle_features],
        }

    @staticmethod
    def from_dict(data: dict) -> "Features":
        individual_features = IndividualFeatures.from_list(data["individual"])
        interaction_features = [InteractionFeatures.from_list(item) for item in data["interaction"]]
        obstacle_features = [ObstacleFeatures.from_list(item) for item in data["obstacle"]]
        return Features(
            individual_features=individual_features,
            interaction_features=interaction_features,
            obstacle_features=obstacle_features
        )

    @staticmethod
    def from_ndjson(json_data: str) -> "Features":
        data = json.loads(json_data)
        return Features.from_dict(data)

    def to_labeled_features(
        self, 
        cur_pos: Point2D = Point2D.zero(), 
        next_pos: Point2D = Point2D.zero(), 
        cur_vel: Velocity = Velocity.zero(), 
        delta_time: float = 0.0
    ) -> "LabeledFeatures":
        return LabeledFeatures(
            individual_features=self.individual_features,
            interaction_features=self.interaction_features,
            obstacle_features=self.obstacle_features,
            cur_pos=cur_pos,
            next_pos=next_pos,
            cur_vel=cur_vel,
            delta_time=delta_time
        )

@dataclass(frozen=True)
class LabeledFeatures(Features):
    cur_pos: Point2D
    next_pos: Point2D
    cur_vel: Velocity
    delta_time: float

    @staticmethod
    def get_labeled_features(
        person: Person,
        person_id: int,
        frame: Frame,
        next_frame: Frame,
        frame_number: int,
        next_frame_number: int,
        fps: float,
        obstacles: list[Obstacle],
    ) -> "LabeledFeatures":
        next_position = next_frame[person_id].position
        if person.velocity is None or person.goal is None:
            raise ValueError(f'Person {person_id} has either velocity, goal, or both equal to None')
        base_features = Features.get_features(person, person_id, frame, obstacles)
        delta_time = (next_frame_number - frame_number) / fps
        return LabeledFeatures(
            individual_features=base_features.individual_features,
            interaction_features=base_features.interaction_features,
            obstacle_features=base_features.obstacle_features,
            cur_pos=person.position,
            next_pos=next_position,
            cur_vel=person.velocity,
            delta_time=delta_time
        )

    def to_tensor(
        self, 
        device: torch.device | str, 
        dtype: torch.dtype = torch.float32, 
        precision: int = 6
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        individual_tensor, interaction_tensors, obstacle_tensors = super().to_tensor(device, dtype, precision)

        cur_pos_tensor = self.cur_pos.to_tensor(device, dtype, precision)
        next_pos_tensor = self.next_pos.to_tensor(device, dtype, precision)
        cur_vel_tensor = self.cur_vel.to_tensor(device, dtype, precision)
        delta_time_tensor = torch.tensor(self.delta_time, device=device, dtype=dtype)

        features = (individual_tensor, interaction_tensors, obstacle_tensors)
        labels = (cur_pos_tensor, next_pos_tensor, cur_vel_tensor, delta_time_tensor)
        return features, labels

    def to_ndjson(self) -> dict[str, Any]:
        features_json = super().to_ndjson()
        features_json["label"] = {
            "cur_pos": self.cur_pos.to_list(), 
            "next_pos": self.next_pos.to_list(),
            "cur_vel": self.cur_vel.to_list(), 
            "dt": self.delta_time
        }
        return features_json

    @staticmethod
    def from_dict(data: dict) -> "LabeledFeatures":
        base_features = Features.from_dict(data)
        label = data["label"]
        cur_pos = Point2D(x=label["cur_pos"][0], y=label["cur_pos"][1])
        next_pos = Point2D(x=label["next_pos"][0], y=label["next_pos"][1])
        cur_vel = Velocity(x=label["cur_vel"][0], y=label["cur_vel"][1])
        delta_time = float(label['dt'])
        return LabeledFeatures(
            individual_features=base_features.individual_features,
            interaction_features=base_features.interaction_features,
            obstacle_features=base_features.obstacle_features,
            cur_pos=cur_pos,
            next_pos=next_pos,
            cur_vel=cur_vel,
            delta_time=delta_time
        )

    @staticmethod
    def from_ndjson(json_data: str) -> "LabeledFeatures":
        data = json.loads(json_data)
        return LabeledFeatures.from_dict(data)

class FrameFeatures:
    def __init__(self, features: dict[int, Features]):
        self.features = features  # person_id -> Features

    @staticmethod
    def get_frame_features(frame: Frame, obstacles: list[Obstacle]) -> "FrameFeatures":
        frame_features = {}
        for person_id, person in frame.items():
            if person.goal is not None and person.velocity is not None:
                features = Features.get_features(person, person_id, frame, obstacles)
                frame_features[person_id] = features
        return FrameFeatures(features=frame_features)

    @staticmethod
    def get_frame_labeled_features(
        frame: Frame, 
        next_frame: Frame,
        frame_number: int,
        next_frame_number: int,
        fps: float,
        obstacles: list[Obstacle]
    ) -> "FrameFeatures":
        frame_features = {}
        valid_person_ids = frame.keys() & next_frame.keys()
        for person_id in valid_person_ids: 
            person = frame[person_id]
            if person.goal is not None and person.velocity is not None:
                labeled_features = LabeledFeatures.get_labeled_features(
                    person, person_id, frame, next_frame, frame_number, next_frame_number, fps, obstacles
                )
                frame_features[person_id] = labeled_features
        return FrameFeatures(features=frame_features)

    def to_list(self) -> list[Features]:
        return list(self.features.values())

    def to_ndjson(self) -> list[dict]:
        return [
            {"person": person_id, "features": features.to_ndjson()}
            for person_id, features in self.features.items()
        ]

    @staticmethod
    def from_dict(data: dict) -> "FrameFeatures":
        features = {
            item["person"]: (
                LabeledFeatures.from_dict(item["features"]) 
                if "label" in item["features"] 
                else Features.from_dict(item["features"])
            )
            for item in data
        }
        return FrameFeatures(features=features)

    @staticmethod
    def from_ndjson(json_data: str) -> "FrameFeatures":
        data = json.loads(json_data)
        return FrameFeatures.from_dict(data) 

class SceneFeatures:
    def __init__(self, features: dict[int, FrameFeatures]):
        self.features = features # frame_number -> FrameFeatures

    @staticmethod
    def get_scene_features(scene: Scene, print_progress: bool = True) -> "SceneFeatures":
        scene_features = {}
        with ThreadPoolExecutor() as executor:
            future_to_frame = {
                executor.submit(FrameFeatures.get_frame_features, scene.frames[frame_number], scene.obstacles): frame_number
                for frame_number in scene.frames
            }
            if print_progress:
                futures = tqdm(future_to_frame, desc="Processing frames")
            else:
                futures = future_to_frame
            for future in futures:
                frame_number = future_to_frame[future]
                scene_features[frame_number] = future.result()
        return SceneFeatures(features=scene_features)

    @staticmethod
    def get_scene_labeled_features(scene: Scene, print_progress: bool = True) -> "SceneFeatures":
        scene_features = {}
        frame_numbers = list(scene.frames.keys())
        with ThreadPoolExecutor() as executor:
            future_to_frame = {
                executor.submit(
                    FrameFeatures.get_frame_labeled_features,
                    scene.frames[frame_numbers[frame_id]],
                    scene.frames[frame_numbers[frame_id + 1]],
                    frame_numbers[frame_id],
                    frame_numbers[frame_id + 1],
                    scene.fps,
                    scene.obstacles
                ): frame_numbers[frame_id]
                for frame_id in range(len(frame_numbers) - 1)  # Exclude the last frame as it has no next frame
            }
            if print_progress:
                futures = tqdm(future_to_frame, desc="Processing frames")
            else:
                futures = future_to_frame
            for future in futures:
                frame_number = future_to_frame[future]
                scene_features[frame_number] = future.result()
        return SceneFeatures(features=scene_features)


    def to_list(self) -> list[Features]:
        with ThreadPoolExecutor() as executor:
            list_of_features = list(executor.map(lambda x: x.to_list(), self.features.values()))
        return [feature for sublist in list_of_features for feature in sublist]

    def to_ndjson(self) -> list[dict]:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda item: [
                    {"frame_number": item[0], "person": f["person"], "features": f["features"]}
                    for f in item[1].to_ndjson()
                ],
                self.features.items()
            ))
        return [item for sublist in results for item in sublist]

    @staticmethod
    def from_dict(data: list[dict], print_progress: bool = True) -> "SceneFeatures":
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item["frame_number"]].append(item)

        with ThreadPoolExecutor() as executor:
            tasks = [(frame_number, frame_data) for frame_number, frame_data in grouped_data.items()]
            if print_progress:
                results = list(tqdm(executor.map(
                    lambda x: (x[0], FrameFeatures.from_dict(x[1])),
                    tasks),
                    total=len(tasks),
                    desc="Processing frames"
                ))
            else:
                results = list(executor.map(
                    lambda x: (x[0], FrameFeatures.from_dict(x[1])),
                    tasks)
                )
        features = dict(results)
        return SceneFeatures(features=features)

    @staticmethod
    def from_ndjson(json_data: str) -> "SceneFeatures":
        data = json.loads(json_data)
        return SceneFeatures.from_dict(data)

def closest_point_on_line(point: Point2D, line_start: Point2D, line_end: Point2D, eps: float = 1e-6) -> Point2D:
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_len = line_vec.magnitude()
    line_unit_vec = line_vec * (1 / (line_len + eps))
    projection_length = point_vec.dot(line_unit_vec)

    # Clamp projection length to line segment bounds [0, line_len]
    projection_length = max(0, min(line_len, projection_length))
    closest_point = line_start + line_unit_vec * projection_length
    
    return closest_point