from typing import List, Tuple
from pedestrian_dataset import PedestrianDataset
from predictor_base import PredictorBase, Predictions
import math

class SocialForcePredictor(PredictorBase):
    def __init__(self, A: float = 2.0, B: float = 0.2, tau: float = 0.5):
        self.A = A  # Interaction force constant
        self.B = B  # Interaction decay constant
        self.tau = tau  # Relaxation time constant

    def calculate_velocity(self, track1: "PedestrianDataset.Record", track2: "PedestrianDataset.Record", fps: float) -> Tuple[float, float]:
        time_difference = (track2['f'] - track1['f']) / fps
        if time_difference == 0:
            return (0.0, 0.0)
        
        velocity_x = (track2['x'] - track1['x']) / time_difference
        velocity_y = (track2['y'] - track1['y']) / time_difference
        
        return (velocity_x, velocity_y)

    def desired_force(self, current_pos: Tuple[float, float], desired_pos: Tuple[float, float], 
                      current_vel: Tuple[float, float], desired_speed: float) -> Tuple[float, float]:
        direction_x = desired_pos[0] - current_pos[0]
        direction_y = desired_pos[1] - current_pos[1]
        distance = math.sqrt(direction_x**2 + direction_y**2)
        
        if distance > 0:
            norm_dir_x = direction_x / distance
            norm_dir_y = direction_y / distance
        else:
            norm_dir_x, norm_dir_y = 0.0, 0.0

        desired_velocity_x = norm_dir_x * desired_speed
        desired_velocity_y = norm_dir_y * desired_speed
        
        force_x = (desired_velocity_x - current_vel[0]) / self.tau
        force_y = (desired_velocity_y - current_vel[1]) / self.tau
        
        return (force_x, force_y)

    def interaction_force(self, pos_i: Tuple[float, float], pos_j: Tuple[float, float],
                          radius: float = 2.0) -> Tuple[float, float]:
        delta_x = pos_i[0] - pos_j[0]
        delta_y = pos_i[1] - pos_j[1]
        distance = math.sqrt(delta_x**2 + delta_y**2)
        
        if distance > 0:
            norm_delta_x = delta_x / distance
            norm_delta_y = delta_y / distance
        else:
            norm_delta_x, norm_delta_y = 0.0, 0.0
        
        force_magnitude = self.A * math.exp((radius - distance) / self.B)
        
        return (force_magnitude * norm_delta_x, force_magnitude * norm_delta_y)

    def predict(self, scene: "PedestrianDataset.Scene", desired_speed: float = 1.5) -> Predictions:
        predictions = []
        fps = scene['fps']
        desired_positions = {}
        for p in scene['records'][-1]:
            (target_records[-1]['x'], target_records[-1]['y'])
        starting_positions = {}

        # Loop through the records to compute velocities and forces for each target record
        for i in range(1, len(target_records)):
            current_track = target_records[i]
            previous_track = target_records[i - 1]
            current_position = (current_track['x'], current_track['y'])
            current_velocity = self.calculate_velocity(previous_track, current_track, fps)

            force_desired = self.desired_force(current_position, desired_position, current_velocity, desired_speed)
            
            # Calculate interaction forces with other pedestrians at current frame
            total_interaction_force_x = 0.0
            total_interaction_force_y = 0.0
            current_frame_id = current_track['f']
            for pid, records in scene['records'].items():
                if pid != target_pid and current_frame_id in records:
                    other_track = records[current_frame_id]
                    other_position = (other_track['x'], other_track['y'])
                    interaction_force = self.interaction_force(current_position, other_position)
                    total_interaction_force_x += interaction_force[0]
                    total_interaction_force_y += interaction_force[1]

            total_force_x = force_desired[0] + total_interaction_force_x
            total_force_y = force_desired[1] + total_interaction_force_y

            acceleration_x = total_force_x
            acceleration_y = total_force_y

            time_difference = (current_track['f'] - previous_track['f']) / fps
            new_velocity_x = current_velocity[0] + acceleration_x * time_difference
            new_velocity_y = current_velocity[1] + acceleration_y * time_difference

            predictions.append(((new_velocity_x, new_velocity_y), (total_force_x, total_force_y)))

        return predictions