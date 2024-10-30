from data.pedestrian_dataset import PedestrianDataset
from models.predictor_base import PredictorBase, Prediction
from core.vector2d import Position, Velocity, Acceleration 

class GroundTruthPredictor(PredictorBase):
    def __init__(self):
        pass

    def predict(self, scene: "PedestrianDataset.Scene") -> Prediction:
        predicted_velocities = {}
        predicted_forces = {}
        fps = scene['fps']

        for i in range(1, len(scene['frame_ids'])):
            frame_id = scene['frame_ids'][i]
            prev_frame_id = scene['frame_ids'][i - 1]
            prev_prev_frame_id = scene['frame_ids'][i - 2]
            delta_time = (frame_id - prev_frame_id) / fps

            for pedestrian_id in scene['person_ids']:
                id = (frame_id, pedestrian_id)
                prev_id = (prev_frame_id, pedestrian_id)
                prev_prev_id = (prev_prev_frame_id, pedestrian_id)

                if all(pos_id in scene['positions'] for pos_id in [prev_prev_id, prev_id, id]):
                    pos = scene['positions'][id]
                    prev_pos = scene['positions'][prev_id]

                    # Calculate velocity based on ground truth position change
                    prev_velocity = Velocity.from_positions(prev_pos, pos, delta_time)
                    predicted_velocities[prev_id] = prev_velocity

                    # Calculate acceleration if there's enough data
                    if prev_prev_id in predicted_velocities:
                        prev_prev_velocity = predicted_velocities[prev_prev_id]
                        prev_prev_acceleration = Acceleration.from_velocities(prev_prev_velocity, prev_velocity, delta_time)
                        predicted_forces[prev_prev_id] = prev_prev_acceleration 
                    else:
                        # Initial acceleration can be set to zero or another placeholder
                        predicted_forces[prev_prev_id] = Acceleration(0.0, 0.0)

        return Prediction(scene_id=scene['id'], predicted_forces=predicted_forces, predicted_velocities=predicted_velocities)
