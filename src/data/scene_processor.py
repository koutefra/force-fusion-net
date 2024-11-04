from typing import List, Dict, Tuple, Optional
from core.scene import Scene
from core.vector2d import Point2D, Velocity
from core.scene_datapoint import SceneDatapoint, SceneDatapoints

class SceneProcessor:
    PERSON_FEATURES_DIM = 8
    INTERACTION_FEATURES_DIM = (None, 5)
    OBSTACLE_FEATURES_DIM = (None, 3)
    LABEL_DIM = 2
    
    def __init__(self, include_focus_persons_only: bool = True):
        self._include_focus_persons_only = include_focus_persons_only
            
    def get_datapoints(self, scenes: Dict[int, Scene]) -> Dict[int, SceneDatapoints]:
        return self._process_scenes(scenes)

    def get_datapoints_from_scene(self, scene: Scene) -> SceneDatapoints:
        return self._process_scene(scene)

    def get_datapoint_from_position(self, scene: Scene, frame_id: int, person_id: int) -> Optional[SceneDatapoint]:
        return self._process_valid_position(scene, frame_id, person_id)

    def get_all_valid_positions(self, scenes: Dict[int, Scene]) -> Dict[Tuple[int, int, int], bool]:
        valid_indices = self._get_valid_positions_all_scenes(scenes)  # List[Tuple[i, i, i]]
        lookup = {key: True for key in valid_indices}
        return lookup

    # private methods
    def _process_scenes(self, scenes: Dict[int, Scene]) -> Dict[int, SceneDatapoints]:
        datapoints = {}
        for scene_id, scene in scenes.items():
            datapoints[scene_id] = self._process_scene(scene)
        return datapoints

    def _process_scene(self, scene: Scene) -> SceneDatapoints:
        datapoints = {}
        valid_positions = self._get_valid_positions_scene(scene)
        for frame_id, person_id in valid_positions:
            datapoints[frame_id] = {}
            focus_persons_ids_pool = scene.focus_person_ids if self._include_focus_persons_only else scene.person_ids
            focus_persons_ids_in_frame = [p_id for p_id in focus_persons_ids_pool if p_id in scene.trajectories[frame_id]]
            for person_id in focus_persons_ids_in_frame:
                datapoints[frame_id][person_id] = self._process_valid_position(scene, frame_id, person_id)
        return datapoints

    def _process_valid_position(self, scene: Scene, frame_id: int, person_id: int) -> SceneDatapoint:
        # get attributes
        frame_positions = scene.trajectories[frame_id]
        frame_obstacles = scene.obstacles[frame_id]
        frame_velocities = scene.velocities[frame_id]
        person_position = frame_positions[person_id]
        person_velocity = frame_velocities[person_id]
        person_acceleration = scene.accelerations[frame_id][person_id]
        goal_position = scene.focus_person_goals[person_id]
        
        # compute features
        person_features = self._compute_person_features(person_position, person_velocity, goal_position)
        interaction_features = self._compute_interaction_features(
            person_position, person_velocity, frame_positions, frame_velocities, person_id
        )
        obstacle_features = self._compute_obstacle_features(person_position, frame_obstacles)
        label = {"acceleration_x": person_acceleration.x, "acceleration_y": person_acceleration.y}
        datapoint = {
            "person_features": person_features,
            "interaction_features": interaction_features,
            "obstacle_features": obstacle_features,
            "label": label
        }
        return datapoint

    def _is_position_valid(self, scene: Scene, frame_id: int, person_id: int) -> bool:
        trajectories = scene.trajectories
        obstacles = scene.obstacles
        velocities = scene.velocities
        accelerations = scene.accelerations

        if person_id not in scene.focus_person_goals or (person_id not in scene.focus_person_ids and self._include_focus_persons_only):
            return False

        if not all(frame_id in attribute for attribute in [trajectories, obstacles, velocities, accelerations]):
            return False

        if not all(person_id in attribute[frame_id] for attribute in [trajectories, velocities, accelerations]):
            return False

        return True

    def _get_valid_positions_scene(self, scene: Scene) -> List[Tuple[int, int]]:
        valid_positions = []
        for frame_id, frame_data in scene.trajectories.items():
            for person_id in frame_data.keys():
                if self._is_position_valid(scene, frame_id, person_id):
                    valid_positions.append((frame_id, person_id))
        return valid_positions

    def _get_valid_positions_all_scenes(self, scenes: Dict[int, Scene]) -> List[Tuple[int, int, int]]:
        valid_positions = []
        for scene_id, scene in scenes.items():
            scene_valid_positions = self._get_valid_positions_scene(scene)
            scene_valid_positions_with_id = map(lambda x: (scene_id, x[0], x[1]), scene_valid_positions)
            valid_positions.extend(scene_valid_positions_with_id)
        return valid_positions
        
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
                                      person_id: int) -> List[Dict[str, float]]:
        interaction_features = []
        for other_id, other_position in frame_positions.items():
            if other_id == person_id:
                continue
            distance = (person_position - other_position).magnitude()
            direction_vector = person_position.direction_to(other_position)
            other_velocity = frame_velocities.get(other_id, Point2D.zero())
            relative_velocity = (person_velocity - other_velocity).magnitude()
            alignment = person_velocity.dot(direction_vector)
            interaction_features.append({
                f"distance_to_person_{other_id}": distance,
                f"direction_x_to_person_{other_id}": direction_vector.x,
                f"direction_y_to_person_{other_id}": direction_vector.y,
                f"relative_velocity_to_person_{other_id}": relative_velocity,
                f"alignment_to_person_{other_id}": alignment,
            })
        return interaction_features

    def _compute_obstacle_features(self, person_position: Point2D, frame_obstacles: List[Point2D]) -> List[Dict[str, float]]:
        obstacle_features = []
        for i, obstacle_position in enumerate(frame_obstacles):
            distance = (person_position - obstacle_position).magnitude()
            direction_vector = person_position.direction_to(obstacle_position)
            obstacle_features.append({
                f"distance_to_obstacle_{i}": distance,
                f"direction_x_to_obstacle_{i}": direction_vector.x,
                f"direction_y_to_obstacle_{i}": direction_vector.y,
            })
        return obstacle_features