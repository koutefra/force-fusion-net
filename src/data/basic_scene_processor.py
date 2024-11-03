from typing import List, Dict, Tuple, Optional
from core.scene import Scene
from core.vector2d import Point2D, Velocity
from core.scene_datapoint import SceneDatapoint, SceneDatapoints
from data.scene_processor import SceneProcessor

class BasicSceneProcessor(SceneProcessor):
    def __init__(self, scenes: Dict[int, Scene], include_focus_persons_only: bool = True):
        super().__init__(scenes)
        self.include_focus_persons_only = include_focus_persons_only





    def _compute_person_features(self, person_position: Point2D, person_velocity: Velocity,
                                 goal_position: Point2D) -> Dict[str, float]:
        distance_to_goal = (person_position - goal_position).magnitude()
        direction_to_goal = person_position.direction_to(goal_position)
        direction_to_goal_normalized = direction_to_goal.normalize()
        velocity_towards_goal = person_velocity.dot(direction_to_goal_normalized)
        return {
            "position_x": person_position.x,
            "position_y": person_position.y,
            "velocity_x": person_velocity.x,
            "velocity_y": person_velocity.y,
            "distance_to_goal": distance_to_goal,
            "direction_x_to_goal": direction_to_goal.x,
            "direction_y_to_goal": direction_to_goal.y,
            "velocity_towards_goal": velocity_towards_goal
        }

    def _compute_interaction_features(self, person_position: Point2D, person_velocity: Velocity, 
                                      frame_positions: Dict[int, Point2D], frame_velocities: Dict[int, Velocity], 
                                      person_id: int) -> Dict[str, float]:
        interaction_features = {}
        for other_id, other_position in frame_positions.items():
            if other_id == person_id:
                continue
            distance = (person_position - other_position).magnitude()
            direction_vector = person_position.direction_to(other_position)
            other_velocity = frame_velocities.get(other_id, Point2D.zero())
            relative_velocity = (person_velocity - other_velocity).magnitude()
            alignment = person_velocity.dot(direction_vector)
            interaction_features.update({
                f"distance_to_person_{other_id}": distance,
                f"direction_x_to_person_{other_id}": direction_vector.x,
                f"direction_y_to_person_{other_id}": direction_vector.y,
                f"relative_velocity_to_person_{other_id}": relative_velocity,
                f"alignment_to_person_{other_id}": alignment,
            })
        return interaction_features

    def _compute_obstacle_features(self, person_position: Point2D, frame_obstacles: List[Point2D]) -> Dict[str, float]:
        obstacle_features = {}
        for i, obstacle_position in enumerate(frame_obstacles):
            distance = (person_position - obstacle_position).magnitude()
            direction_vector = person_position.direction_to(obstacle_position)
            obstacle_features.update({
                f"distance_to_obstacle_{i}": distance,
                f"direction_x_to_obstacle_{i}": direction_vector.x,
                f"direction_y_to_obstacle_{i}": direction_vector.y,
            })
        return obstacle_features

    def process_position(self, scene: Scene, frame_id: int, person_id: int) -> Optional[Tuple[Dict[str, float], Dict[str, float]]]:
        required_attributes = [
            scene.trajectories, 
            scene.obstacles, 
            scene.velocities_central_difference, 
            scene.accelerations_central_difference
        ]

        if not all(frame_id in attribute for attribute in required_attributes):
            return None

        frame_positions = scene.trajectories[frame_id]
        frame_obstacles = scene.obstacles[frame_id]
        frame_velocities = scene.velocities_central_difference[frame_id]
        person_position = frame_positions[person_id]
        person_velocity = frame_velocities[person_id]
        person_acceleration = scene.accelerations_central_difference[frame_id][person_id]
        goal_position = scene.focus_person_goals[person_id]
        
        person_features = self._compute_person_features(person_position, goal_position)
        interaction_features = self._compute_interaction_features(
            person_position, person_velocity, frame_positions, frame_velocities, person_id
        )
        obstacle_features = self._compute_obstacle_features(person_position, frame_obstacles)

        features = person_features.copy()
        features.update(interaction_features)
        features.update(obstacle_features)

        label = {"acceleration_x": person_acceleration.x, "acceleration_y": person_acceleration.y}

        return features, label

    def process_scene(self, scene: Scene) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        datapoints = ...
        for frame_id in scene.sorted_frame_ids:
            datapoints[frame_id] = {}
            focus_persons_ids_pool = scene.focus_person_ids if self.include_focus_persons_only else scene.person_ids
            focus_persons_ids_in_frame = [p_id for p_id in focus_persons_ids_pool if p_id in scene.trajectories[frame_id]]
            for person_id in focus_persons_ids_in_frame:
                datapoint = self.position_to_datapoints(scene, frame_id, person_id)
                if datapoint:
                    datapoints[frame_id][person_id] = datapoint
        return datapoints

    def index_to_feature(feature_type: str, index: int) -> str:
        reference_list = None

        if feature_type == "person":
            reference_list = person_features
        elif feature_type == "interaction":
            reference_list = interaction_features
        elif obstacle_features == "obstacle":
            reference_list = obstacle_features

        if not reference_list:
            raise ValueError(f"Feature type {feature_type} not implemented.")

        return reference_list[index]
            



    person_features = ["position_x", "position_y", "velocity_x", "velocity_y", "distance_to_goal", 
                        "direction_x_to_goal", "direction_y_to_goal"]
    interaction_features = ["distance_to_person_p", "direction_x_to_person_p", "direction_y_to_person_p",
                            "relative_velocity_to_person_p", "alignment_to_person_p"]
    obstacle_features = ["distance_to_obstacle_o", "direction_x_to_obstacle_o", "direction_y_to_obstacle_o"]