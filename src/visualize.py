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
from evaluation.angle_animator import AngleCosineAnimator
from entities.vector2d import Point2D
from entities.scene import Scene
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

parser.add_argument("--angle_anim", action="store_true",
                    help="Open split-view angle animator (or save MP4 if headless).")
parser.add_argument("--person_id", type=int, default=None, help="Person ID to animate.")
parser.add_argument("--refx", type=float, default=0.0, help="Reference point X.")
parser.add_argument("--refy", type=float, default=0.0, help="Reference point Y.")
parser.add_argument("--show_components", action="store_true",
                    help="If set, plot decomposed force cosines when available.")
parser.add_argument("--scene_json", type=str, default=None,
                    help="Optional: load precomputed Scene JSON (with forces).")
parser.add_argument("--save_angle_mp4", type=str, default=None,
                    help="If set, save angle animation to this MP4 path (also used automatically when headless).")


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Option A: load a precomputed scene (may include forces/decomposition)
    if args.scene_json:
        scene = Scene.from_json(args.scene_json)
    else:
        # load dataset
        if args.dataset_type == 'juelich_bneck':
            dataset = SceneDataset.from_loaders([JulichCaserneLoader([(args.dataset_path, args.dataset_name)])])
        else:
            raise ValueError(f"Unknown dataset type: {args.dataset_type}")

        # velocities
        dataset = dataset.approximate_velocities(args.fdm_win_size, "backward")
        scene = next(iter(dataset.scenes.values()))

        # optional predictor
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

        # simulate or trim
        if args.predictor_type != 'gt' and predictor is not None:
            scene = scene.simulate(
                predict_acc_func=predictor.predict,
                total_steps=args.animation_steps,
                goal_radius=args.goal_radius
            )
        else:
            scene = scene.take_first_n_frames(args.animation_steps)

        # smooth fields
        scene = scene.approximate_velocities(args.fdm_win_size, "central")
        scene = scene.approximate_accelerations(args.fdm_win_size, "central")

    # Angle animator mode (interactive if possible; MP4 otherwise)
    if args.angle_anim:
        pid = args.person_id or (sorted(scene.get_all_person_ids())[0])
        ref = Point2D(args.refx, args.refy)
        AngleCosineAnimator(scene, ref_point=ref).animate_person(
            person_id=pid,
            show_components=args.show_components,
            save_path=args.save_angle_mp4,
            arrow_scale=1.5
        )
        return

    # Your original flow if not using angle animator
    Visualizer.plot_trajectories(scene, output_file_path=f'results/trajectories_{args.predictor_type}_{scene.id}.png')
    if not args.create_plot_only:
        Animation().create(scene, draw_person_ids=args.draw_person_ids, time_scale=args.time_scale, desc=args.predictor_type)


if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))