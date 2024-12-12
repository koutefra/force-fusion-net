import argparse
import numpy as np
from data.scene_dataset import SceneDataset
from entities.scene import Scene
from data.loaders.juelich_bneck_loader import JuelichBneckLoader 
from data.fdm_calculator import FiniteDifferenceCalculator
from visualization.visualization import Visualizer
from models.direct_net import DirectNet
from models.social_force import SocialForce
from models.predictor import Predictor
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", required=True, type=str, help="The dataset type.")
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--dataset_name", required=True, type=str, help="The dataset name.")
parser.add_argument("--predictor_path", required=False, type=str, help="The trained model paths.")
parser.add_argument("--predictor_type",  type=str, default="gt", help="The trained model type.")
parser.add_argument("--fdm_win_size", default=20, type=int, help="Finitie difference method window size.")
parser.add_argument("--time_scale", default=1.0, type=float, help="Time scale of the animations.")
parser.add_argument("--sampling_step", default=1, type=int, help="Dataset sampling step.")
parser.add_argument("--animation_steps", default=300, type=int, help="How many steps should be simulated.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device to use (e.g., 'cpu', 'cuda').")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    if args.dataset_type == 'juelich_bneck':
        fdm_calculator = FiniteDifferenceCalculator(args.fdm_win_size)
        dataset = SceneDataset([JuelichBneckLoader(
            [(args.dataset_path, args.dataset_name)], 
            args.sampling_step, 
            compute_accelerations=True, 
            fdm_calculator=fdm_calculator)
        ])
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # load predictor
    predictor = None
    if args.predictor_path:
        if args.predictor_type != 'gt':
            if args.predictor_type == 'neural_net':
                model = DirectNet.from_weight_file(args.predictor_path)
            elif args.predictor_type == 'social_force':
                model = SocialForce.from_weight_file(args.predictor_path)
            else:
                raise ValueError(f"Unknown predictor type: {args.predictor_type}")

            predictor = Predictor(model, device=args.device)
        
    visualizer = Visualizer()
    scene = next(iter(dataset.scenes.values()))

    if args.predictor_type != 'gt':
        scene = scene.simulate(
            predict_acc_func=predictor.predict,
            total_steps=args.animation_steps,
            goal_radius=0.5
        )
    else:
        scene = scene.take_first_n_frames(args.animation_steps)

    visualizer.visualize(scene, time_scale=args.time_scale, desc=args.predictor_type)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))