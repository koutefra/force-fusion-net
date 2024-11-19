from dataclasses import dataclass, asdict
from collections import defaultdict
from entities.scene import Scene, BaseObstacle, Frame, Person
from entities.vector2d import Point2D
from entities.obstacle import PointObstacle, LineObstacle
import torch
from typing import Any
import json

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
    velocity_x: float
    velocity_y: float
    distance_to_goal: float
    direction_x_to_goal: float
    direction_y_to_goal: float
    velocity_towards_goal: float

    @staticmethod
    def get_individual_features(person: Person, goal_pos: Point2D) -> "IndividualFeatures":
        dist_to_goal = (person.position - goal_pos).magnitude()
        dir_to_goal = person.position.direction_to(goal_pos)
        dir_to_goal_norm = dir_to_goal.normalize()
        vel_towards_goal = person.velocity.dot(dir_to_goal_norm)
        return IndividualFeatures(
            velocity_x=person.velocity.x,
            velocity_y=person.velocity.y,
            distance_to_goal=dist_to_goal,
            direction_x_to_goal=dir_to_goal.x,
            direction_y_to_goal=dir_to_goal.y,
            velocity_towards_goal=vel_towards_goal
        )

@dataclass(frozen=True)
class InteractionFeatures(FeatureBase):
    distance_to_person_p: float
    direction_x_to_person_p: float
    direction_y_to_person_p: float
    relative_velocity_to_person_p: float
    alignment_to_person_p: float

    @staticmethod
    def get_interaction_features(person: Person, person_id: int, frame: Frame) -> list["InteractionFeatures"]:
        interaction_features = []
        for other_person_id, other_person in frame.items():
            if person_id == other_person_id:
                continue
            
            distance = (person.position - other_person.position).magnitude()
            direction_vector = person.position.direction_to(other_person.position)
            relative_velocity = (person.velocity - other_person.velocity).magnitude()
            alignment = person.velocity.dot(direction_vector)
            
            interaction_features.append(InteractionFeatures(
                distance_to_person_p=distance,
                direction_x_to_person_p=direction_vector.x,
                direction_y_to_person_p=direction_vector.y,
                relative_velocity_to_person_p=relative_velocity,
                alignment_to_person_p=alignment,
            ))
        return interaction_features

@dataclass(frozen=True)
class ObstacleFeatures(FeatureBase):
    distance_to_obstacle_o: float
    direction_x_to_obstacle_o: float
    direction_y_to_obstacle_o: float
    
    @staticmethod
    def get_obstacle_features(person: Person, obstacles: list[BaseObstacle]) -> list["ObstacleFeatures"]:
        obstacle_features = []
        for obstacle in obstacles:
            if isinstance(obstacle, PointObstacle):
                distance = (person.position - obstacle.position).magnitude()
                direction_vector = person.position.direction_to(obstacle.position)
            elif isinstance(obstacle, LineObstacle):
                closest_point = closest_point_on_line(person.position, obstacle.line[0], obstacle.line[1])
                distance = (person.position - closest_point).magnitude()
                direction_vector = person.position.direction_to(closest_point)

            obstacle_features.append(ObstacleFeatures(
                distance_to_obstacle_o=distance,
                direction_x_to_obstacle_o=direction_vector.x,
                direction_y_to_obstacle_o=direction_vector.y,
            ))
        return obstacle_features

@dataclass(frozen=True)
class Features:
    individual_features: IndividualFeatures
    interaction_features: list[InteractionFeatures]
    obstacle_features: list[ObstacleFeatures]

    @staticmethod
    def get_features(person: Person, person_id: int, frame: Frame, obstacles: list[BaseObstacle]) -> "Features":
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

@dataclass(frozen=True)
class LabeledFeatures(Features):
    cur_pos: Point2D
    next_pos: Point2D

    @staticmethod
    def get_labeled_features(
        person: Person,
        person_id: int,
        frame: Frame,
        next_frame: Frame,
        obstacles: list[BaseObstacle],
    ) -> "LabeledFeatures":
        next_position = next_frame[person_id].position
        base_features = Features.get_features(person, person_id, frame, obstacles)
        return LabeledFeatures(
            individual_features=base_features.individual_features,
            interaction_features=base_features.interaction_features,
            obstacle_features=base_features.obstacle_features,
            cur_pos=person.position,
            next_pos=next_position
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
        return (individual_tensor, interaction_tensors, obstacle_tensors), (cur_pos_tensor, next_pos_tensor)

    def to_ndjson(self) -> dict[str, Any]:
        features_json = super().to_ndjson()
        features_json["label"] = {"cur_pos": self.cur_pos.to_list(), "next_pos": self.next_pos.to_list()}
        return features_json

    @staticmethod
    def from_dict(data: dict) -> "LabeledFeatures":
        base_features = Features.from_dict(data)
        label = data["label"]
        cur_pos = Point2D(x=label["cur_pos"][0], y=data["cur_pos"][1])
        next_pos = Point2D(x=label["next_pos"][0], y=data["next_pos"][1])
        return LabeledFeatures(
            individual_features=base_features.individual_features,
            interaction_features=base_features.interaction_features,
            obstacle_features=base_features.obstacle_features,
            cur_pos=cur_pos,
            next_pos=next_pos
        )

    @staticmethod
    def from_ndjson(json_data: str) -> "LabeledFeatures":
        data = json.loads(json_data)
        return LabeledFeatures.from_dict(data)

class FrameFeatures:
    def __init__(self, features: dict[int, Features]):
        self.features = features  # person_id -> Features

    @staticmethod
    def get_frame_features(frame: Frame, obstacles: list[BaseObstacle]) -> "FrameFeatures":
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
        obstacles: list[BaseObstacle]
    ) -> "FrameFeatures":
        frame_features = {}
        valid_person_ids = frame.keys() & next_frame.keys()
        for person_id in valid_person_ids: 
            person = frame[person_id]
            if person.goal is not None and person.velocity is not None:
                labeled_features = LabeledFeatures.get_labeled_features(
                    person, person_id, frame, next_frame, obstacles
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
    def get_scene_features(scene: Scene) -> "SceneFeatures":
        scene_features = {}
        for frame_number, frame in scene.frames.items():
            frame_features = FrameFeatures.get_frame_features(frame, scene.obstacles)
            scene_features[frame_number] = frame_features
        return SceneFeatures(features=scene_features)

    @staticmethod
    def get_scene_labeled_features(scene: Scene) -> "SceneFeatures":
        scene_features = {}
        frame_numbers = list(scene.frames.keys())
        for frame_id in range(len(frame_numbers) - 1):  # Stop at the second-to-last frame
            frame_number = frame_numbers[frame_id]
            next_frame_number = frame_numbers[frame_id + 1]
            frame = scene.frames[frame_number]
            next_frame = scene.frames[next_frame_number]
            frame_features = FrameFeatures.get_frame_labeled_features(frame, next_frame, scene.obstacles)
            scene_features[frame_number] = frame_features
        return SceneFeatures(features=scene_features)

    def to_list(self) -> list[Features]:
        return [feature for frame_features in self.features.values() for feature in frame_features.to_list()]

    def to_ndjson(self) -> list[dict]:
        return [
            {
                "frame_number": frame_number,
                "person": f_features["person"],
                "features": f_features["features"]
            }
            for frame_number, frame_features in self.features.items()
            for f_features in frame_features.to_ndjson()
        ]

    @staticmethod
    def from_dict(data: list[dict]) -> "SceneFeatures":
        grouped_data = defaultdict(list)
        for item in data:
            grouped_data[item["frame_number"]].append(item)

        features = {
            frame_number: FrameFeatures.from_dict(frame_data)
            for frame_number, frame_data in grouped_data.items()
        }

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