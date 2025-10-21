import argparse
import json
from pathlib import Path
from evaluation.evaluator import Evaluator
from evaluation.flow_plotter import FlowPlotter
from utils.io_utils import set_seed, load_scene, load_model

# ==========================================================
# Flow rectangles for each bottleneck width (approximation)
# ==========================================================
FLOW_RECTS = {
    "synth1": [1.0, 3.0, -0.8, 0.8],
    "b090": [1.0, 3.0, -0.45, 0.45],
    "b100": [1.0, 3.0, -0.5, 0.5],
    "b110": [1.0, 3.0, -0.55, 0.55],
    "b120": [1.0, 3.0, -0.6, 0.6],
    "b140": [1.0, 3.0, -0.7, 0.7],
    "b160": [1.0, 3.0, -0.8, 0.8],
    "b180": [1.0, 3.0, -0.9, 0.9],
    "b200": [1.0, 3.0, -1.0, 1.0],
    "b220": [1.0, 3.0, -1.1, 1.1],
    "b250": [1.0, 3.0, -1.25, 1.25],
    "l0":   [1.0, 3.0, -0.6, 0.6],
    "l2":   [1.0, 3.0, -0.6, 0.6],
    "l4":   [1.0, 3.0, -0.6, 0.6],
}


def main(args):
    # ------------------ SEED CONTROL ------------------
    set_seed(args.seed)

    # ------------------ SCENE ------------------
    scene, dataset = load_scene(args.dataset_folder, args.scene_file, args.scene_name, args.fdm_win_size)
    scene_name = scene.id
    flow_rect = FLOW_RECTS.get(scene_name, [1.0, 3.0, -0.8, 0.8]) if args.flow_rect is None else tuple(args.flow_rect)
    flow_axis = tuple(args.flow_axis)

    # ------------------ OUTPUT DIRECTORY ------------------
    out_dir = Path(args.out_dir) / scene_name / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator()
    results = {}

    # ------------------ SIMULATION ------------------
    if args.model_type == "gt":
        scene = scene.take_first_n_frames(args.simulation_steps)
        scene = scene.approximate_velocities(args.fdm_win_size, "central")
        scene = scene.approximate_accelerations(args.fdm_win_size, "central")
    else:
        predictor = load_model(args.model_folder, args.model_file, args.model_type, args.device)
        print(f"[INFO] Running simulation for {scene_name} ({args.model_type}) ...")
        scene = scene.simulate(
            predict_acc_func=predictor.predict,
            total_steps=args.simulation_steps,
            x_threshold=scene.bounding_box[1][0] - args.x_goal_offset,
        )

    # ------------------ EVALUATION ------------------
    results.update(evaluator.evaluate_scene(scene))
    results.update(evaluator.evaluate_force_magnitudes(scene))
    results.update(evaluator.evaluate_min_distances(scene))

    # ADE / FDE vs ground truth
    scene_gt = next(iter(dataset.scenes.values())).take_first_n_frames(args.simulation_steps)
    results.update(evaluator.evaluate_individual_ADE_FDE(scene_gt, scene))
    results.update(evaluator.evaluate_flow_curve(scene, flow_rect=flow_rect, flow_axis=flow_axis))

    # ------------------ SAVE RESULTS ------------------
    plot_path = out_dir / f"flow_curve_{args.model_type}_{scene_name}.png"
    FlowPlotter.plot_flow_curve(results, out_path=plot_path, title=f"Flow curve – {args.model_type} on {scene_name}")
    print(f"✅ Flow curve plot saved to {plot_path}")

    metrics_path = out_dir / f"metrics_{args.model_type}_{scene_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a single bottleneck scene.")
    parser.add_argument("--dataset_folder", default="./data/datasets/julich_bottleneck_caserne", type=str)
    parser.add_argument("--scene_file", required=True, type=str)
    parser.add_argument("--scene_name", required=False, type=str)
    parser.add_argument("--model_folder", default="./data/weights", type=str)
    parser.add_argument("--model_file", required=True, type=str)
    parser.add_argument("--model_type", required=True, choices=["fusion_net", "direct_net", "social_force", "social_force_b160", "gt"])
    parser.add_argument("--fdm_win_size", type=int, default=20)
    parser.add_argument("--simulation_steps", type=int, default=5000)
    parser.add_argument("--flow_rect", type=float, nargs=4, default=None)
    parser.add_argument("--flow_axis", type=float, nargs=2, default=[1.0, 0.0])
    parser.add_argument("--x_goal_offset", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out_dir", type=str, default="./results")

    main(parser.parse_args())
