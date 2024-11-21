import argparse
import numpy as np
from data.scene_dataset import SceneDataset
from data.loaders.trajnet_loader import TrajnetLoader
from data.loaders.juelich_bneck_loader import JuelichBneckLoader 
from visualization.visualization import Visualizer
from models.neural_net_predictor import NeuralNetPredictor
from models.neural_net_model import NeuralNetModel
from models.social_force_model import SocialForceModel
from models.social_force_predictor import SocialForcePredictor
from data.fdm_calculator import FiniteDifferenceCalculator
import random
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--dataset_type", required=True, type=str, help="The dataset type.")
parser.add_argument("--dataset_name", required=True, type=str, help="The dataset name.")
parser.add_argument("--predictor_path", required=False, type=str, help="The trained model paths.")
parser.add_argument("--predictor_type",  type=str, help="The trained model type.")
parser.add_argument("--scenes_to_show", default=[1], nargs='*', type=int, help="IDs of scenes to visualize. If None, random scenes are selected.")
parser.add_argument("--sampling_step", default=5, type=int, help="Sampling step.")
parser.add_argument("--fdm_win_size", default=20, type=int, help="Finitie difference method window size.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    if args.dataset_type == 'juelich_bneck':
        fdm_calculator = FiniteDifferenceCalculator(args.fdm_win_size)
        loader = JuelichBneckLoader([(args.dataset_path, args.dataset_name)], args.sampling_step, fdm_calculator)
        scene_dataset = SceneDataset({'juelich_bneck': loader})
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # load predictor
    predictor = None
    if args.predictor_path:
        if args.predictor_type == 'neural_net':
            model = NeuralNetModel.from_weight_file(args.predictor_path)
            predictor = NeuralNetPredictor(model)
        elif args.predictor_type == 'social_force':
            with open(args.predictor_path, "r") as file:
                param_grid = json.load(file)
            delta_time = 0.4
            model = SocialForceModel(delta_time=delta_time, **param_grid)
            predictor = SocialForcePredictor(model)
        else:
            raise ValueError(f"Unknown predictor type: {args.predictor_type}")
        
    visualizer = Visualizer()
    scene_id = "b090"
    scene = scene_dataset.scenes['juelich_bneck'][scene_id]
    scene_transformed = scene.simulate(
        predict_acc_func=predictor.predict,
        frame_step=args.sampling_step,
        total_steps=100
    )

    visualizer.visualize(scene_transformed, time_scale=1.0/args.sampling_step)
    # visualizer.visualize(scene, time_scale=1/args.sampling_step)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))