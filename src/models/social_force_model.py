from data.scene_collection import PedestrianDataset
from core.vector2d import Point2D, Velocity, Acceleration 
from core.scene import Scene
import math
from typing import Dict, List, Tuple

class SocialForcePredictor:
    def __init__(self, A: float = 1.5, B: float = 0.4, tau: float = 1.0, radius: float = 1.5, 
                 desired_speed: float = 1.5):
        self.A = A  # Interaction force constant
        self.B = B  # Interaction decay constant
        self.radius = radius  # Interaction radius
        self.tau = tau  # Relaxation time constant
        self.desired_speed = desired_speed

    def _desired_force(self, cur_point: Point2D, desired_point: Point2D, velocity: Velocity) -> Acceleration:
        direction = desired_point - cur_point 
        desired_velocity = direction.normalize() * self.desired_speed
        desired_acceleration = (desired_velocity - velocity) * (1 / self.tau)
        return Acceleration(desired_acceleration.x, desired_acceleration.y)

    def _interaction_force(self, point_i: Point2D, point_j: Point2D) -> Acceleration:
        direction = point_i - point_j
        distance = direction.magnitude()
        force_magnitude = self.A * math.exp((self.radius - distance) / self.B)
        interaction_acceleration = direction.normalize() * force_magnitude
        return Acceleration(interaction_acceleration.x, interaction_acceleration.y)

    def _obstacle_force(self, point: Point2D, obstacle: Point2D) -> Acceleration:
        return self._interaction_force(point, obstacle)

    def _compute_interaction_forces(self, positions: Dict[int, Point2D], person_id: int) -> Acceleration:
        """Compute the sum of interaction forces from all other pedestrians."""
        total_interaction_force_x = 0.0
        total_interaction_force_y = 0.0
        person_pos = positions[person_id]
        for other_person_pos in positions.keys():
            if person_pos == other_person_pos:
                continue
            interaction_force = self._interaction_force(person_pos, other_person_pos)
            total_interaction_force_x += interaction_force.x
            total_interaction_force_y += interaction_force.y
        return Acceleration(x=total_interaction_force_x, y=total_interaction_force_y)

    def predict_scene(self, scene: Scene) -> Dict[int, Dict[int, Acceleration]]:
        predicted_accelerations = {}
        velocities = scene.velocities_central_difference

        for frame_id, person_positions in scene.trajectories.items():
            predicted_accelerations[frame_id] = {}
            for person_id in scene.focus_person_ids:
                if person_id not in person_positions:
                    continue

                current_position = person_positions[person_id]
                goal_position = scene.focus_person_goals[person_id]
                velocity = velocities[frame_id][person_id]
                force_desired = self._desired_force(current_position, goal_position, velocity)
                total_interaction_force = self._compute_interaction_forces(person_positions, person_id)
                total_force = force_desired + total_interaction_force
                predicted_accelerations[frame_id][person_id] = Acceleration(x=total_force.x, y=total_force.y)
        return predicted_accelerations

    def predict(self, scenes: Dict[Scene]) -> Dict[Dict[int, Dict[int, Acceleration]]]:
        predicted_accelerations = {}
        for scene_id, scene in scenes.items():
            predicted_accelerations[scene_id] = self.predict_scene(scene)
        return predicted_accelerations