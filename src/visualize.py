import argparse
import numpy as np
from data.scene_collection import SceneCollection
from data.trajnet_loader import TrajnetLoader
from visualization.visualization import Visualizer
from models.neural_net_predictor import NeuralNetPredictor
from models.social_force_predictor import SocialForcePredictor
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--dataset_type", default='trajnet++', type=str, help="The dataset type.")
parser.add_argument("--predictor_path", required=False, type=str, help="The trained model paths.")
parser.add_argument("--predictor_type",  type=str, help="The trained model type.")
parser.add_argument("--scenes_to_show", required=False, default=[], nargs='*', type=str, help="IDs of scenes to visualize. If None, random scenes are selected.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    if args.dataset_type == "trajnet++":
        loader = TrajnetLoader(args.dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    scene_collection = SceneCollection([loader])

    if len(args.scenes_to_show) == 0:
        n_scenes = len(scene_collection.scenes.keys())
        n_samples = min(5, n_scenes)
        args.scenes_to_show = random.sample(list(scene_collection.scenes.keys()), n_samples)

    # load predictor and get predictions
    predictor = None
    if args.predictor_path:
        if args.predictor_type == 'neural_net':
            predictor = NeuralNetPredictor(args.predictor_path)
        elif args.predictor_type == 'social_force':
            predictor = SocialForcePredictor(args.predictor_path)
        else:
            raise ValueError(f"Unknown predictor type: {args.predictor_type}")
        
    # run visualization
    visualizer = Visualizer()
    for scene_id in args.scenes_to_show:
        scene = scene_collection.scenes[scene_id]

        pred_accelerations = predictor.predict_scene(scene)
        
        preds = [(args.predictor_type, pred_accelerations, Visualizer.default_colors["skin_orange"])] 

        visualizer.visualize(scene, preds=preds)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))