import argparse
from pathlib import Path
from utils.io_utils import set_seed, load_scene, load_model
from evaluation.evaluator import Evaluator


def main(args):
    # ------------------ SEED CONTROL ------------------
    set_seed(args.seed)

    # ------------------ LOAD SCENE ------------------
    scene, _ = load_scene(args.dataset_folder, args.scene_file, args.scene_name, args.fdm_win_size)

    # ------------------ LOAD MODEL ------------------
    predictor = load_model(args.model_folder, args.model_file, args.model_type, args.device)

    # ------------------ COMPUTE PREDICTED FORCES ------------------
    print(f"[INFO] Computing predicted forces for {scene.id} using {args.model_type}")

    # Resolve output path
    if args.out_path:
        out_path = Path(args.out_path)
        if out_path.is_dir():
            out_path = out_path / f"{scene.id}_forces_{args.model_type}.json"
    else:
        out_path = Path(f"scene_{scene.id}_with_forces_{args.model_type}.json")

    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute and save
    Evaluator().compute_predicted_forces(scene, predictor).to_json(str(out_path))
    print(f"✅ Predicted forces saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute predicted forces for each frame in a scene.")

    # Dataset
    parser.add_argument("--dataset_folder", default="./data/datasets/julich_bottleneck_caserne", type=str)
    parser.add_argument("--scene_file", required=True, type=str)
    parser.add_argument("--scene_name", required=False, type=str)

    # Model
    parser.add_argument("--model_folder", default="./data/weights", type=str)
    parser.add_argument("--model_file", required=True, type=str)
    parser.add_argument("--model_type", required=True, type=str,
                        choices=["fusion_net", "direct_net", "social_force", "social_force_b160"])

    # Misc
    parser.add_argument("--fdm_win_size", type=int, default=20)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out_path", type=str, default="./data/forces",
                        help="Optional custom output path (file or directory).")

    main(parser.parse_args())
