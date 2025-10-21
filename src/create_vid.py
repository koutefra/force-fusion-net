import argparse
from pathlib import Path
from utils.io_utils import set_seed, load_scene, load_model
from evaluation.visualizer import Visualizer
from evaluation.video_maker import VideoMaker

def main(args):
    # ------------------ SEED ------------------
    set_seed(args.seed)

    # ------------------ SCENE ------------------
    scene, _ = load_scene(args.dataset_folder, args.scene_file, args.scene_name, args.fdm_win_size)
    scene_name = scene.id

    # ------------------ OUTPUT DIRECTORY ------------------
    out_dir = Path(args.out_dir) / scene_name / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------ MODEL ------------------
    predictor = None
    if args.model_type != "gt":
        predictor = load_model(args.model_folder, args.model_file, args.model_type, args.device)
        print(f"[INFO] Simulating {args.simulation_steps} steps using {args.model_type} ...")
        scene = scene.simulate(
            predict_acc_func=predictor.predict,
            total_steps=args.simulation_steps,
            x_threshold=scene.bounding_box[1][0] - args.x_goal_offset,
        )
    else:
        print(f"[INFO] Using ground truth data (first {args.simulation_steps} frames).")
        scene = scene.take_first_n_frames(args.simulation_steps)
        scene = scene.approximate_velocities(args.fdm_win_size, "central")
        scene = scene.approximate_accelerations(args.fdm_win_size, "central")

    # ------------------ VISUALIZATION ------------------
    img_path = out_dir / f"trajectories.png"
    Visualizer.plot_trajectories(scene, output_file_path=str(img_path))
    print(f"[INFO] Trajectory plot saved to {img_path}")

    if not args.create_plot_only:
        print(f"[INFO] Generating animation...")
        VideoMaker(output_dir=out_dir).create(
            scene,
            draw_person_ids=args.draw_person_ids,
            time_scale=args.time_scale,
            desc=args.model_type,
        )
        print("[INFO] Video creation completed successfully.")

    print("\n✅ Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and animate pedestrian flow simulations.")
    parser.add_argument("--dataset_folder", default="./data/datasets/julich_bottleneck_caserne", type=str)
    parser.add_argument("--scene_file", required=True, type=str)
    parser.add_argument("--scene_name", required=False, type=str)
    parser.add_argument("--model_folder", default="./data/weights", type=str)
    parser.add_argument("--model_file", required=True, type=str)
    parser.add_argument("--model_type", required=True, choices=["fusion_net", "direct_net", "social_force", "social_force_b160", "gt"])
    parser.add_argument("--fdm_win_size", default=20, type=int)
    parser.add_argument("--simulation_steps", default=300, type=int)
    parser.add_argument("--time_scale", default=2.0, type=float)
    parser.add_argument("--x_goal_offset", type=float, default=1.0)
    parser.add_argument("--create_plot_only", action="store_true")
    parser.add_argument("--draw_person_ids", action="store_true")
    parser.add_argument("--seed", default=21, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--out_dir", type=str, default="./results")

    main(parser.parse_args())
