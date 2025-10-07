import argparse
import numpy as np
import random
import torch
from data.scene_dataset import SceneDataset
from data.julich_caserne_loader import JulichCaserneLoader
from evaluation.evaluator import Evaluator
from models.predictor import Predictor
from models.fusion_net import FusionNet
from models.direct_net import DirectNet
from models.social_force import SocialForce
from models.social_force_b160 import SocialForceB160
from evaluation.flow_plotter import FlowPlotter


def main(args):
    # Seed control
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load scene
    dataset = SceneDataset.from_loaders([
        JulichCaserneLoader([(args.dataset_path, args.scene_name)])
    ])
    dataset = dataset.approximate_velocities(args.fdm_win_size, "backward")
    scene = next(iter(dataset.scenes.values()))

    # Evaluate
    flow_rect = tuple(args.flow_rect)   # (xmin, xmax, ymin, ymax)
    flow_axis = tuple(args.flow_axis)   # (ax, ay)
    evaluator = Evaluator(flow_rect=flow_rect, flow_axis=flow_axis)
    results = {}

    # Simulate
    if args.model_type == "gt":
        # Just trim the ground truth to match the time span of predicted sim
        scene = scene.take_first_n_frames(args.simulation_steps)
        scene = scene.approximate_accelerations(args.fdm_win_size, "central")
    else:
        # Load model
        if args.model_type == "fusion_net":
            model = FusionNet.from_weight_file(args.model_path)
        elif args.model_type == "direct_net":
            model = DirectNet.from_weight_file(args.model_path)
        elif args.model_type == "social_force":
            model = SocialForce.from_weight_file(args.model_path)
        elif args.model_type == "social_force_b160":
            model = SocialForceB160.from_weight_file(args.model_path)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

        predictor = Predictor(model, device=args.device)

        if args.precompute_forces:
            evaluator.compute_predicted_forces(scene, predictor).to_json(f'scene_{scene.id}_with_forces_{args.model_type}.json')
            return

        scene = scene.simulate(
            predict_acc_func=predictor.predict,
            total_steps=args.simulation_steps,
            goal_radius=args.goal_radius
        )

    # 1. Classic frame-based evaluation (collisions, CoM, velocities, etc.)
    results.update(evaluator.evaluate_scene(scene))

    # 2. Force magnitude statistics
    results.update(evaluator.evaluate_force_magnitudes(scene))

    # 3. Minimum distance to other pedestrians
    results.update(evaluator.evaluate_min_distances(scene))

    # 4. ADE / FDE per pedestrian
    scene_gt = next(iter(dataset.scenes.values())).take_first_n_frames(args.simulation_steps)
    results.update(evaluator.evaluate_individual_ADE_FDE(scene_gt, scene))

    # 5. Flow curve evaluation
    results.update(evaluator.evaluate_flow_curve(scene))

    plot_file = f"flow_curve_{args.model_type}_{scene.id}.png"
    FlowPlotter.plot_flow_curve(
        results,
        out_path=plot_file,
        title=f"Flow curve – {args.model_type} on {scene.id}"
    )
    print(f"\nFlow curve plot saved to {plot_file}")

    # Output
    print(f"\n📊 Evaluation for {args.model_type} on scene {scene.id}:\n" + "-" * 60)
    for k, v in results.items():
        print(f"{k:25s}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--scene_name", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--model_type", required=True, type=str,
                        choices=["fusion_net", "direct_net", "social_force", "social_force_b160", "gt"])
    parser.add_argument("--fdm_win_size", type=int, default=20)
    parser.add_argument("--simulation_steps", type=int, default=2000)
    parser.add_argument("--goal_radius", type=float, default=0.5)
    parser.add_argument(
        "--flow_rect", type=float, nargs=4,
        default=[1.0, 3.0, -0.8, 0.8],
        help="Measurement rectangle (xmin xmax ymin ymax), default: scene b160"
    )
    parser.add_argument(
        "--flow_axis", type=float, nargs=2,
        default=[1.0, 0.0],
        help="Flow direction axis (ax ay), default: right direction"
    )
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument(
        "--precompute_forces",
        action="store_true",
        help="If set, precompute forces for all frames instead of running full simulation"
    )
    parser.add_argument("--seed", type=int, default=21)

    main(parser.parse_args())
