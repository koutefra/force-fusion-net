from typing import List, Tuple
from pedestrian_dataset import PedestrianDataset
from predictor_base import PredictorBase, Prediction, Velocity, Force
import math

class SocialForcePredictor(PredictorBase):
    def __init__(self, A: float = 2.0, B: float = 0.2, tau: float = 0.5):
        self.A = A  # Interaction force constant
        self.B = B  # Interaction decay constant
        self.tau = tau  # Relaxation time constant

    def calculate_velocity(self, track1: "PedestrianDataset.Pos", track2: "PedestrianDataset.Pos",
                           time_difference: float) -> Velocity:
        if time_difference == 0:
            return (0.0, 0.0)
        
        velocity_x = (track2['x'] - track1['x']) / time_difference
        velocity_y = (track2['y'] - track1['y']) / time_difference
        
        return Velocity(vx=velocity_x, vy=velocity_y)

    def desired_force(self, pos: PedestrianDataset.Pos, desired_pos: PedestrianDataset.Pos, 
                      velocity: Velocity, desired_speed: float) -> Force:
        direction_x = desired_pos['x'] - pos['x']
        direction_y = desired_pos['y'] - pos['y']
        distance = math.sqrt(direction_x**2 + direction_y**2)
        
        if distance > 0:
            norm_dir_x = direction_x / distance
            norm_dir_y = direction_y / distance
        else:
            norm_dir_x, norm_dir_y = 0.0, 0.0

        desired_velocity_x = norm_dir_x * desired_speed
        desired_velocity_y = norm_dir_y * desired_speed
        
        force_x = (desired_velocity_x - velocity['vx']) / self.tau
        force_y = (desired_velocity_y - velocity['vy']) / self.tau
        
        return Force(fx=force_x, fy=force_y)

    def interaction_force(self, pos_i: Tuple[float, float], pos_j: Tuple[float, float],
                          radius: float = 2.0) -> Tuple[float, float]:
        delta_x = pos_i['x'] - pos_j['x']
        delta_y = pos_i['y'] - pos_j['y']
        distance = math.sqrt(delta_x**2 + delta_y**2)
        
        if distance > 0:
            norm_delta_x = delta_x / distance
            norm_delta_y = delta_y / distance
        else:
            norm_delta_x, norm_delta_y = 0.0, 0.0
        
        force_magnitude = self.A * math.exp((radius - distance) / self.B)
        
        return (force_magnitude * norm_delta_x, force_magnitude * norm_delta_y)

    def predict(self, scene: "PedestrianDataset.Scene", desired_speed: float = 1.5) -> Prediction:
        preds = {}
        fps = scene['fps']

        # Loop through the records to compute velocities and forces for each target record
        for level in range(1, len(scene['frame_numbers'])):
            frame_number = scene['frame_numbers'][level]
            prev_frame_number = scene['frame_numbers'][level - 1]
            time_difference = (frame_number - prev_frame_number) / fps

            for pedestrian_id in scene['pedestrian_ids']:
                id = (frame_number, pedestrian_id) 
                prev_id = (prev_frame_number, pedestrian_id)

                if prev_id not in scene['positions']:
                    continue

                pos = scene['positions'][id]
                prev_pos = scene['positions'][prev_id]
                desired_pos_id = (scene['end_frames'][pedestrian_id], pedestrian_id)
                desired_pos = scene['positions'][desired_pos_id]

                velocity = self.calculate_velocity(prev_pos, pos, time_difference)

                force_desired = self.desired_force(pos, desired_pos, velocity, desired_speed)
                
                # Calculate interaction forces with other pedestrians at current frame
                total_interaction_force_x = 0.0
                total_interaction_force_y = 0.0
                for other_pedestrian_id in scene['pedestrian_ids']:
                    other_pos_id = (frame_number, other_pedestrian_id)
                    if other_pedestrian_id != pedestrian_id and other_pos_id in scene['positions']:
                        other_pos = scene['positions'][other_pos_id]
                        interaction_force = self.interaction_force(pos, other_pos)
                        total_interaction_force_x += interaction_force[0]
                        total_interaction_force_y += interaction_force[1]

                total_force_x = force_desired['fx'] + total_interaction_force_x
                total_force_y = force_desired['fy'] + total_interaction_force_y

                acceleration_x = total_force_x
                acceleration_y = total_force_y

                new_velocity_x = velocity['vx'] + acceleration_x * time_difference
                new_velocity_y = velocity['vy'] + acceleration_y * time_difference

                preds[id] = (Velocity(vx=new_velocity_x, vy=new_velocity_y), 
                             Force(fx=total_force_x, fy=total_force_y))

        return Prediction(scene_id=scene['id'], preds=preds)