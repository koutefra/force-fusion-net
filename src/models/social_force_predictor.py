from data.pedestrian_dataset import PedestrianDataset
from models.predictor_base import PredictorBase, Prediction
from core.vector2d import Position, Velocity, Acceleration 
import math

class SocialForcePredictor(PredictorBase):
    def __init__(self, A: float = 1.5, B: float = 0.4, tau: float = 1.0, radius: float = 1.5, 
                 desired_speed: float = 1.5):
        self.A = A  # Interaction force constant
        self.B = B  # Interaction decay constant
        self.radius = radius  # Interaction radius
        self.tau = tau  # Relaxation time constant
        self.desired_speed = desired_speed

    def desired_force(self, pos: Position, desired_pos: Position, velocity: Velocity) -> Acceleration:
        direction = desired_pos - pos
        desired_velocity = direction.normalize() * self.desired_speed
        desired_acceleration = (desired_velocity - velocity) * (1 / self.tau)
        return Acceleration(desired_acceleration.x, desired_acceleration.y)

    def interaction_force(self, pos_i: Position, pos_j: Position) -> Acceleration:
        direction = pos_i - pos_j
        distance = direction.magnitude()
        force_magnitude = self.A * math.exp((self.radius - distance) / self.B)
        interaction_acceleration = direction.normalize() * force_magnitude
        return Acceleration(interaction_acceleration.x, interaction_acceleration.y)

    def compute_interaction_forces(self, pos: Position, pedestrian_id: int, scene: "PedestrianDataset.Scene", 
                                   frame_id: int) -> Acceleration:
        """Compute the sum of interaction forces from all other pedestrians."""
        total_interaction_force_x = 0.0
        total_interaction_force_y = 0.0
        for other_pedestrian_id in scene['person_ids']:
            if other_pedestrian_id == pedestrian_id:
                continue
            other_pos_id = (frame_id, other_pedestrian_id)
            if other_pos_id in scene['positions']:
                other_pos = scene['positions'][other_pos_id]
                interaction_force = self.interaction_force(pos, other_pos)
                total_interaction_force_x += interaction_force.x
                total_interaction_force_y += interaction_force.y
        return Acceleration(x=total_interaction_force_x, y=total_interaction_force_y)

    def predict(self, scene: "PedestrianDataset.Scene") -> Prediction:
        predicted_forces = {}
        predicted_velocities = {}
        fps = scene['fps']

        # Loop through the records to compute velocities and forces for each target record
        for level in range(1, len(scene['frame_ids'])):
            frame_id = scene['frame_ids'][level]
            prev_frame_id = scene['frame_ids'][level - 1]
            delta_time = (frame_id - prev_frame_id) / fps

            for pedestrian_id in scene['person_ids']:
                id = (frame_id, pedestrian_id) 
                prev_id = (prev_frame_id, pedestrian_id)

                if prev_id not in scene['positions'] or id not in scene['positions']:
                    continue

                pos = scene['positions'][id]
                prev_pos = scene['positions'][prev_id]
                desired_pos_id = (scene['end_frames'][pedestrian_id], pedestrian_id)
                desired_pos = scene['positions'][desired_pos_id]

                velocity = Velocity.from_positions(prev_pos, pos, delta_time)
                predicted_velocities[id] = velocity

                force_desired = self.desired_force(pos, desired_pos, velocity)
                total_interaction_force = self.compute_interaction_forces(pos, pedestrian_id, scene, frame_id)

                total_force = force_desired + total_interaction_force
                predicted_forces[id] = Acceleration(x=total_force.x, y=total_force.y)

        return Prediction(scene_id=scene['id'], predicted_forces=predicted_forces, predicted_velocities=predicted_velocities)