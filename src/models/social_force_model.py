from data.dataset import PedestrianDataset
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

    def desired_force(self, cur_point: Point2D, desired_point: Point2D, velocity: Velocity) -> Acceleration:
        direction = desired_point - cur_point 
        desired_velocity = direction.normalize() * self.desired_speed
        desired_acceleration = (desired_velocity - velocity) * (1 / self.tau)
        return Acceleration(desired_acceleration.x, desired_acceleration.y)

    def interaction_force(self, point_i: Point2D, point_j: Point2D) -> Acceleration:
        direction = point_i - point_j
        distance = direction.magnitude()
        force_magnitude = self.A * math.exp((self.radius - distance) / self.B)
        interaction_acceleration = direction.normalize() * force_magnitude
        return Acceleration(interaction_acceleration.x, interaction_acceleration.y)

    def obstacle_force(self, point: Point2D, obstacle: Point2D) -> Acceleration:
        return self.interaction_force(point, obstacle)

    def compute_interaction_forces(self, positions: Dict[int, Point2D], person_id: int) -> Acceleration:
        """Compute the sum of interaction forces from all other pedestrians."""
        total_interaction_force_x = 0.0
        total_interaction_force_y = 0.0
        person_pos = positions[person_id]
        for other_person_pos in positions.keys():
            if person_pos == other_person_pos:
                continue
            interaction_force = self.interaction_force(person_pos, other_person_pos)
            total_interaction_force_x += interaction_force.x
            total_interaction_force_y += interaction_force.y
        return Acceleration(x=total_interaction_force_x, y=total_interaction_force_y)

    def predict_scene(self, scene: Scene) -> Dict[int, Dict[int, Acceleration]]:
        predicted_accelerations = {}
        fps = scene['fps']

        # Question. Use `scene` or `datapoints` (dict) object?
        raise NotImplementedError()

    def predict(self, scenes: Dict[Scene]) -> Dict[Dict[int, Dict[int, Acceleration]]]:
        raise NotImplementedError()
        

        # for level in range(1, len(scene['frame_ids'])):
        #     frame_id = scene['frame_ids'][level]
        #     prev_frame_id = scene['frame_ids'][level - 1]
        #     delta_time = (frame_id - prev_frame_id) / fps

        #     for pedestrian_id in scene['person_ids']:
        #         id = (frame_id, pedestrian_id) 
        #         prev_id = (prev_frame_id, pedestrian_id)

        #         if prev_id not in scene['positions'] or id not in scene['positions']:
        #             continue

        #         pos = scene['positions'][id]
        #         prev_pos = scene['positions'][prev_id]
        #         desired_pos_id = (scene['end_frames'][pedestrian_id], pedestrian_id)
        #         desired_pos = scene['positions'][desired_pos_id]

        #         velocity = Velocity.from_positions(prev_pos, pos, delta_time)
        #         predicted_velocities[id] = velocity

        #         force_desired = self.desired_force(pos, desired_pos, velocity)
        #         total_interaction_force = self.compute_interaction_forces(pos, pedestrian_id, scene, frame_id)

        #         total_force = force_desired + total_interaction_force
        #         predicted_forces[id] = Acceleration(x=total_force.x, y=total_force.y)

        # return Prediction(scene_id=scene['id'], predicted_forces=predicted_forces, predicted_velocities=predicted_velocities)