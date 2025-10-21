#!/usr/bin/env python3
import argparse
from pathlib import Path
from entities.vector2d import Point2D
from evaluation.visualizer import Visualizer
from utils.io_utils import parse_models_arg


def main(args: argparse.Namespace) -> None:
    ref = Point2D(args.refx, args.refy)

    # --- Mode A: Multi-model comparison ---
    if args.compare_models:
        models = parse_models_arg(args.compare_models)
        # Try to infer scene name from the first provided path
        first_path = next(iter(models.values())).id if hasattr(next(iter(models.values())), "id") else "unknown"
        inferred_name = first_path or "comparison"

        out_dir = Path(args.out_dir or f"./results/force_viz_{inferred_name}")
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Saving comparison plots to: {out_dir}")

        if args.person_id is not None:
            out_path = out_dir / f"pid_{args.person_id:03d}.png"
            Visualizer.plot_force_comparison_2x5(
                models=models,
                person_id=args.person_id,
                ref_point=ref,
                out_path=str(out_path),
                show_components=not args.no_components,
                dpi=args.dpi,
            )
        else:
            Visualizer.plot_force_comparison_batch_2x5(
                models=models,
                person_id=None,
                ref_point=ref,
                out_dir=str(out_dir),
                show_components=not args.no_components,
                dpi=args.dpi,
            )
        return

    # --- Mode B: Single-scene panels ---
    if not args.scene_json:
        raise ValueError("Provide either --compare_models or --scene_json.")

    from entities.scene import Scene
    scene = Scene.from_json(args.scene_json)
    model_name = Path(args.scene_json).stem
    models = {model_name: scene}

    # Dynamically infer output directory name
    scene_name = getattr(scene, "id", None) or model_name
    out_dir = Path(args.out_dir or f"./results/force_viz_{scene_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Saving visualizations to: {out_dir}")

    if args.person_id is not None:
        out_path = out_dir / f"pid_{args.person_id:03d}.png"
        Visualizer.plot_force_comparison_2x5(
            models=models,
            person_id=args.person_id,
            ref_point=ref,
            out_path=str(out_path),
            show_components=not args.no_components,
            dpi=args.dpi,
        )
    else:
        Visualizer.plot_force_comparison_batch_2x5(
            models=models,
            person_id=None,
            ref_point=ref,
            out_dir=str(out_dir),
            show_components=not args.no_components,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force visualization: 2×5 panels and multi-model comparison.")
    parser.add_argument("--refx", type=float, default=0.0)
    parser.add_argument("--refy", type=float, default=0.0)
    parser.add_argument("--person_id", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (defaults to results/force_viz_{scene_name})")
    parser.add_argument("--no_components", action="store_true")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--compare_models", type=str, default=None,
                        help="Comma-separated NAME:path.json pairs, e.g. 'FFN:a.json,FDN:b.json,SFM:c.json'.")
    parser.add_argument("--scene_json", type=str, default=None)
    main(parser.parse_args())
