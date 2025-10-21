#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import copy

from utils.io_utils import set_seed, load_scene, load_model
from evaluation.evaluator import Evaluator
from evaluation.visualizer import Visualizer


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading base scene {args.scene_file}...")
    base_scene, _ = load_scene(args.dataset_folder, args.scene_file, args.scene_name, args.fdm_win_size)
    print(f"[INFO] Scene loaded with {len(base_scene.frames)} frames.")

    thresholds = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    evaluator = Evaluator()
    model_specs = [m.strip() for m in args.models.split(",") if m.strip()]
    coll_results = {}

    for spec in model_specs:
        if ":" not in spec:
            raise ValueError(f"Malformed model entry: {spec} (expected TYPE:path)")
        model_type, model_path = spec.split(":", 1)
        model_type, model_path = model_type.strip(), model_path.strip()

        print(f"\n🧩 Processing model: {model_type}")
        scene = copy.deepcopy(base_scene)

        if model_type == "gt" or model_path == "-":
            print(f"[INFO] Using ground truth for {args.scene_name}")
            scene = scene.take_first_n_frames(args.simulation_steps)
            scene = scene.approximate_velocities(args.fdm_win_size, "central")
            scene = scene.approximate_accelerations(args.fdm_win_size, "central")
        else:
            predictor = load_model(
                model_folder=args.model_folder,
                model_file=model_path if "/" not in model_path else Path(model_path).name,
                model_type=model_type,
                device=args.device,
            )
            scene = scene.simulate(
                predict_acc_func=predictor.predict,
                total_steps=args.simulation_steps,
                x_threshold=scene.bounding_box[1][0] - args.x_goal_offset,
            )

        coll_results[model_type] = evaluator.evaluate_collision_vs_threshold(scene, thresholds)

    print("\n📈 Plotting agent collisions vs threshold...")
    Visualizer.plot_collision_vs_threshold_multi(
        data_dict=coll_results,
        key="agent_collisions",
        title=f"Agent Collisions vs Threshold ({args.scene_name})",
        out_path=str(out_dir / f"agent_collisions_vs_threshold_{args.scene_name}.png"),
    )

    print("📉 Plotting obstacle collisions vs threshold...")
    Visualizer.plot_collision_vs_threshold_multi(
        data_dict=coll_results,
        key="obstacle_collisions",
        title=f"Obstacle Collisions vs Threshold ({args.scene_name})",
        out_path=str(out_dir / f"obstacle_collisions_vs_threshold_{args.scene_name}.png"),
    )

    print(f"\n✅ All plots saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare collision–threshold curves across multiple models.")
    parser.add_argument("--dataset_folder", default="./data/datasets/julich_bottleneck_caserne", type=str)
    parser.add_argument("--scene_file", required=True, type=str)
    parser.add_argument("--scene_name", required=False, type=str)
    parser.add_argument("--models", required=True, type=str)
    parser.add_argument("--model_folder", default="./data/weights", type=str)
    parser.add_argument("--fdm_win_size", type=int, default=20)
    parser.add_argument("--simulation_steps", type=int, default=4000)
    parser.add_argument("--x_goal_offset", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--thr_min", type=float, default=0.05)
    parser.add_argument("--thr_max", type=float, default=0.3)
    parser.add_argument("--thr_step", type=float, default=0.025)
    parser.add_argument("--out_dir", type=str, default="results/collision_comparison")
    main(parser.parse_args())
