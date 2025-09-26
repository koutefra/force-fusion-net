import argparse
import numpy as np
from data.scene_dataset import SceneDataset
from data.julich_caserne_loader import JulichCaserneLoader 
from evaluation.animation import Animation
from models.direct_net import DirectNet
from models.fusion_net import FusionNet
from models.social_force import SocialForce
from models.social_force_b160 import SocialForceB160
from models.predictor import Predictor
from evaluation.visualizer import Visualizer
import random

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_type", required=True, type=str, help="The dataset type.")
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--dataset_name", required=True, type=str, help="The dataset name.")
parser.add_argument("--predictor_path", required=False, type=str, help="The trained model paths.")
parser.add_argument("--predictor_type",  type=str, default="gt", help="The trained model type.")
parser.add_argument("--fdm_win_size", default=20, type=int, help="Finitie difference method window size.")
parser.add_argument("--time_scale", default=2.0, type=float, help="Time scale of the animations.")
parser.add_argument("--sampling_step", default=1, type=int, help="Dataset sampling step.")
parser.add_argument("--animation_steps", default=300, type=int, help="How many steps should be simulated.")
parser.add_argument("--create_plot_only", action="store_true", help="Creates only plot, no animation.")
parser.add_argument("--draw_person_ids", action="store_true", help="Draw person IDs in the middle of the circle.")
parser.add_argument("--goal_radius", default=0.6, type=float, help="The radius around goal positions.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device to use (e.g., 'cpu', 'cuda').")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    if args.dataset_type == 'juelich_bneck':
        dataset = SceneDataset.from_loaders([JulichCaserneLoader([(args.dataset_path, args.dataset_name)])])
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # appxoximate quantities
    dataset = dataset.approximate_velocities(args.fdm_win_size, "backward")
    
    # load predictor
    predictor = None
    if args.predictor_path and args.predictor_type != 'gt':
        if args.predictor_type == 'direct_net':
            model = DirectNet.from_weight_file(args.predictor_path)
        elif args.predictor_type == 'fusion_net':
            model = FusionNet.from_weight_file(args.predictor_path)
        elif args.predictor_type == 'social_force':
            model = SocialForce.from_weight_file(args.predictor_path)
        elif args.predictor_type == 'social_force_b160':
            model = SocialForceB160.from_weight_file(args.predictor_path)
        else:
            raise ValueError(f"Unknown predictor type: {args.predictor_type}")

        predictor = Predictor(model, device=args.device)

    scene = next(iter(dataset.scenes.values()))

    if args.predictor_type != 'gt':
        scene = scene.simulate(
            predict_acc_func=predictor.predict,
            total_steps=args.animation_steps,
            goal_radius=args.goal_radius
        )
    else:
        scene = scene.take_first_n_frames(args.animation_steps)

    scene = scene.approximate_velocities(args.fdm_win_size, "central")
    scene = scene.approximate_accelerations(args.fdm_win_size, "central")

    Visualizer.plot_trajectories(scene, output_file_path=f'results/trajectories_{args.predictor_type}_{scene.id}.png')

    if not args.create_plot_only:
        Animation().create(scene, draw_person_ids=args.draw_person_ids, time_scale=args.time_scale, desc=args.predictor_type)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))