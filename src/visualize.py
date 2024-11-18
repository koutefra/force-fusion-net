import argparse
import numpy as np
from data.scene_dataset import SceneDataset
from data.loaders.trajnet_loader import TrajnetLoader
from data.loaders.juelich_bneck_loader import JuelichBneckLoader 
from data.feature_extractor import FeatureExtractor
from visualization.visualization import Visualizer
from models.neural_net_predictor import NeuralNetPredictor
from models.social_force_predictor import SocialForcePredictor
from data.fdm_calculator import FiniteDifferenceCalculator
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--dataset_type", required=True, type=str, help="The dataset type.")
parser.add_argument("--dataset_name", required=True, type=str, help="The dataset name.")
parser.add_argument("--predictor_path", required=False, type=str, help="The trained model paths.")
parser.add_argument("--predictor_type",  type=str, help="The trained model type.")
parser.add_argument("--scenes_to_show", default=[1], nargs='*', type=int, help="IDs of scenes to visualize. If None, random scenes are selected.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    loaders = {}
    if args.dataset_type == "trajnet++":
        fdm_calculator = FiniteDifferenceCalculator(win_size=2)
        loader = TrajnetLoader(args.dataset_path, args.dataset_name, fdm_calculator)
    elif args.dataset_type == "juelich_bneck":
        fdm_calculator = FiniteDifferenceCalculator(win_size=20)
        loader = JuelichBneckLoader(args.dataset_path, args.dataset_name, fdm_calculator)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    loaders[args.dataset_name] = loader
    feature_extractor = FeatureExtractor()
    scene_dataset = SceneDataset(loaders, feature_extractor, load_on_demand=True)

    # load predictor
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
    for scene_id in args.scenes_to_show:
        scene = scene_dataset.get_scene(args.dataset_name, scene_id)

        preds = []
        # if predictor:
            # new_trajectory = scene.simulate_trajectory(predictor.predict_frame)
            # preds = [(args.predictor_type, new_trajectory, Visualizer.default_colors["skin_orange"])] 

        visualizer.visualize(scene, time_scale=1.0)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))