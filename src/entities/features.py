from dataclasses import dataclass, asdict
from entities.scene import Scene, BaseObstacle, Frame, Person
from entities.vector2d import Point2D
from entities.obstacle import PointObstacle, LineObstacle
import torch
from typing import Any
import json
import pickle

@dataclass(frozen=True)
class FeatureBase:
    def to_list(self) -> list[float]:
        return list(asdict(self).values())

    @classmethod
    def dim(cls) -> int:
        return len(cls.__dataclass_fields__)

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.to_list(), dtype=torch.float32)

    def to_json(self) -> dict:
        return asdict(self)

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

    def to_tensor(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        individual_tensor = self.individual_features.to_tensor()
        interaction_tensors = (
            torch.stack([feat.to_tensor() for feat in self.interaction_features])
            if self.interaction_features else torch.empty((0, InteractionFeatures.dim()))
        )
        obstacle_tensors = (
            torch.stack([feat.to_tensor() for feat in self.obstacle_features])
            if self.obstacle_features else torch.empty((0, ObstacleFeatures.dim()))
        )
        return individual_tensor, interaction_tensors, obstacle_tensors

    def to_json(self) -> dict[str, Any]:
        return {
            "individual_features": self.individual_features.to_json(),
            "interaction_features": [feat.to_json() for feat in self.interaction_features],
            "obstacle_features": [feat.to_json() for feat in self.obstacle_features],
        }

    @staticmethod
    def from_dict(data: dict) -> "Features":
        individual_features = IndividualFeatures(**data["individual_features"])
        interaction_features = [InteractionFeatures(**item) for item in data["interaction_features"]]
        obstacle_features = [ObstacleFeatures(**item) for item in data["obstacle_features"]]
        
        return Features(
            individual_features=individual_features,
            interaction_features=interaction_features,
            obstacle_features=obstacle_features
        )

class FrameFeatures:
    def __init__(self, features: dict[int, Features]):
        self.features = features  # person_id -> Features

    @staticmethod
    def get_frame_features(frame: Frame, obstacles: list[BaseObstacle]) -> "FrameFeatures":
        frame_features = {}
        for person_id, person in frame.items():
            if person.goal and person.velocity:
                features = Features.get_features(person, person_id, frame, obstacles)
                frame_features[person_id] = features
        return frame_features

    def to_json(self) -> dict[int, dict]:
        return {person_id: feature.to_json() for person_id, feature in self.features.items()}

    @staticmethod
    def from_dict(data: dict) -> "FrameFeatures":
        features = {int(person_id): Features.from_dict(feature_data) for person_id, feature_data in data.items()}
        return FrameFeatures(features=features)

class SceneFeatures:
    def __init__(self, frames: dict[int, FrameFeatures]):
        self.frames = frames  # frame_number -> FrameFeatures

    @staticmethod
    def get_features(scene: Scene) -> "SceneFeatures":
        scene_features = {}
        for frame_number, frame in scene.frames.items():
            frame_features = FrameFeatures.get_frame_features(frame, scene.obstacles)
            scene_features[frame_number] = frame_features
        return scene_features

    def to_json(self) -> dict[int, dict]:
        return {frame_number: frame_features.to_json() for frame_number, frame_features in self.frames.items()}

    @staticmethod
    def from_dict(data: dict) -> "SceneFeatures":
        frames = {int(frame_number): FrameFeatures.from_dict(frame_data) for frame_number, frame_data in data.items()}
        return SceneFeatures(frames=frames)

    @classmethod
    def load(cls, filepath: str, save_format: str = "json") -> "SceneFeatures":
        if save_format == "json":
            with open(f"{filepath}.json", "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        elif save_format == "pickle":
            with open(f"{filepath}.pkl", "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported save format. Use 'json' or 'pickle'.")

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