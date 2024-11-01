from typing import List, Dict
from core.scene import Scene
from core.vector2d import Point2D
from core.scene_datapoint import SceneDatapoint, SceneDatapoints

class SceneFeatureExtractor:
    def __init__(self, include_focus_persons_only: bool = True):
        self.include_focus_persons_only = include_focus_persons_only

    def _compute_interaction_features(self, person_position: Point2D, person_velocity: Point2D, 
                                     frame_positions: Dict[int, Point2D], frame_velocities: Dict[int, Point2D], 
                                     person_id: int) -> Dict[str, float]:
        interaction_features = {}
        for other_id, other_position in frame_positions.items():
            if other_id == person_id:
                continue  # Skip self
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

    def _compute_goal_features(self, person_position: Point2D, goal_position: Point2D) -> Dict[str, float]:
        distance_to_goal = (person_position - goal_position).magnitude()
        direction_to_goal = person_position.direction_to(goal_position)
        return {
            "distance_to_goal": distance_to_goal,
            "direction_x_to_goal": direction_to_goal.x,
            "direction_y_to_goal": direction_to_goal.y
        }

    def position_to_datapoint(self, scene: Scene, frame_id: int, person_id: int) -> SceneDatapoint:
        frame_positions = scene.trajectories[frame_id]
        frame_obstacles = scene.obstacles[frame_id]
        frame_velocities = scene.velocities_central_difference[frame_id]
        person_position = frame_positions[person_id]
        person_velocity = frame_velocities[person_id]
        person_acceleration = scene.accelerations_central_difference[frame_id][person_id]
        
        interaction_features = self._compute_interaction_features(
            person_position, person_velocity, frame_positions, frame_velocities, person_id
        )
        obstacle_features = self._compute_obstacle_features(person_position, frame_obstacles)
        goal_position = scene.focus_person_goals[person_id]
        goal_features = self._compute_goal_features(person_position, goal_position)
        general_features = {
            "position_x": person_position.x,
            "position_y": person_position.y,
            "velocity_x": person_velocity.x,
            "velocity_y": person_velocity.y,
            **goal_features
        }
        label = {"acceleration_x": person_acceleration.x, "acceleration_y": person_acceleration.y}

        scene_datapoint: SceneDatapoint = {
            "person_features": general_features,
            "interaction_features": interaction_features,
            "obstacle_features": obstacle_features,
            "label": label
        }
        return scene_datapoint


    def scene_to_datapoints(self, scene: Scene) -> SceneDatapoints:
        datapoints: SceneDatapoints = {}
        
        for frame_id in scene.sorted_frame_ids:
            datapoints[frame_id] = {}
            focus_persons_ids_pool = scene.focus_person_ids if self.include_focus_persons_only else scene.person_ids
            focus_persons_ids_in_frame = [p_id for p_id in focus_persons_ids_pool if p_id in scene.trajectories[frame_id]]
            for person_id in focus_persons_ids_in_frame:
                datapoints[frame_id][person_id] = self.position_to_datapoint(scene, frame_id, person_id)
        
        return datapoints