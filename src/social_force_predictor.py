from typing import List, Tuple
from pedestrian_dataset import PedestrianDataset
from predictor_base import PredictorBase, Prediction
from vector2d import Position, Velocity, Force
import math

class SocialForcePredictor(PredictorBase):
    def __init__(self, A: float = 2.0, B: float = 0.2, tau: float = 0.5):
        self.A = A  # Interaction force constant
        self.B = B  # Interaction decay constant
        self.tau = tau  # Relaxation time constant

    def calculate_velocity(self, track1: Position, track2: Position, time_difference: float) -> Velocity:
        if time_difference == 0:
            return (0.0, 0.0)

        difference = (track2 - track1) / time_difference
        return Velocity(x=difference.x, y=difference.y)  

    def desired_force(self, pos: Position, desired_pos: Position, velocity: Velocity, 
                      desired_speed: float) -> Force:
        direction = desired_pos - pos
        distance = math.sqrt(direction.x**2 + direction.y**2)
        
        if distance > 0:
            norm_direction = direction / distance
        else:
            norm_direction = Position(x=0.0, y=0.0)

        desired_velocity = norm_direction * desired_speed
        force = (desired_velocity - velocity) / self.tau
        return Force(x=force.x, y=force.y)

    def interaction_force(self, pos_i: Position, pos_j: Position, radius: float = 2.0) -> Force:
        delta_pos = pos_i - pos_j
        distance = math.sqrt(delta_pos.x**2 + delta_pos.y**2)
        
        if distance > 0:
            norm_delta_pos = delta_pos / distance
        else:
            norm_delta_pos = Position(x=0.0, y=0.0)
        
        force_magnitude = self.A * math.exp((radius - distance) / self.B)
        force = force_magnitude * norm_delta_pos
        return Force(x=force.x, y=force.y)

    def predict(self, scene: "PedestrianDataset.Scene", desired_speed: float = 1.5) -> Prediction:
        preds = {}
        fps = scene['fps']


        # dodelat time management, rozdily mezi frame_id a prev_frame_id nemusi byt konstantni
        # fps je pouze relativni 
        # force i velocity by mely byt skalovany spravne


        # Loop through the records to compute velocities and forces for each target record
        for level in range(1, len(scene['frame_ids'])):
            frame_id = scene['frame_ids'][level]
            prev_frame_id = scene['frame_ids'][level - 1]
            time_difference = (frame_id - prev_frame_id) / fps

            for pedestrian_id in scene['person_ids']:
                id = (frame_id, pedestrian_id) 
                prev_id = (prev_frame_id, pedestrian_id)

                if prev_id not in scene['positions'] or id not in scene['positions']:
                    continue

                pos = scene['positions'][id]
                prev_pos = scene['positions'][prev_id]
                desired_pos_id = (scene['end_frames'][pedestrian_id], pedestrian_id)
                desired_pos = scene['positions'][desired_pos_id]

                velocity = self.calculate_velocity(prev_pos, pos, time_difference)
                force_desired = self.desired_force(pos, desired_pos, velocity, desired_speed)
                
                # Calculate interaction forces with other pedestrians at current frame
                total_interaction_force = Force(x=0.0, y=0.0)
                for other_pedestrian_id in scene['person_ids']:
                    other_pos_id = (frame_id, other_pedestrian_id)
                    if other_pedestrian_id != pedestrian_id and other_pos_id in scene['positions']:
                        other_pos = scene['positions'][other_pos_id]
                        interaction_force = self.interaction_force(pos, other_pos)
                        total_interaction_force += interaction_force

                total_force = force_desired + total_interaction_force
                acceleration = total_force
                new_velocity = velocity + acceleration * time_difference

                preds[id] = (Velocity(x=new_velocity.x, y=new_velocity.y), 
                             Force(x=total_force.x, y=total_force.y))

        return Prediction(scene_id=scene['id'], preds=preds)