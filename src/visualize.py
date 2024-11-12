import argparse
import numpy as np
from data.scene_dataset import SceneDataset
from data.trajnet_loader import TrajnetLoader
from entities.frame_object import PersonInFrame
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

    scene_dataset = SceneDataset({"loader_0": loader}, load_on_demand=False)

    # load predictor and get predictions
    predictor = None
    if args.predictor_path:
        if args.predictor_type == 'neural_net':
            predictor = NeuralNetPredictor(args.predictor_path, device='cpu')
        elif args.predictor_type == 'social_force':
            predictor = SocialForcePredictor(args.predictor_path)
        else:
            raise ValueError(f"Unknown predictor type: {args.predictor_type}")
        
    # run visualization
    visualizer = Visualizer()
    # for scene_id in args.scenes_to_show:
    for scene_id in range(50, 55):
        scene = scene_dataset.get_scene("loader_0", scene_id)

        preds = []
        if predictor:
            new_trajectory = scene.simulate_trajectory(predictor.predict_frame)
            preds = [(args.predictor_type, new_trajectory, Visualizer.default_colors["skin_orange"])] 

        visualizer.visualize(scene, preds=preds)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))