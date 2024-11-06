from core.scene import Scene
from typing import Dict, Callable, Any
from core.vector2d import Acceleration

def simulate_scene_force_preds(scene: Scene, 
                               preprocess_func: Callable[[Scene, int, int], Any],
                               predict_func: Callable[[Any], Acceleration]):
    for frame_id, frame_data in scene.trajectories.items():
        for person_id in frame_data.keys():
            preprocessed_scene = preprocess_func(scene, frame_id, person_id)
            if not preprocessed_scene:
                ...
                
            predicted_force = predict_func(preprocessed_scene, frame_id, person_id)
            

