import argparse, os
from pathlib import Path
from typing import Dict
from entities.scene import Scene
from entities.vector2d import Point2D
from evaluation.visualizer import Visualizer

def _parse_models_arg(models_arg: str) -> Dict[str, Scene]:
    """
    Parse 'FFN:path.json,FDN:path.json' into {'FFN': Scene(...), 'FDN': Scene(...)}.
    """
    out: Dict[str, Scene] = {}
    parts = [p.strip() for p in models_arg.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            raise ValueError(f"Malformed --compare_models entry: '{p}' (expected NAME:path.json)")
        name, path = p.split(":", 1)
        name = name.strip(); path = path.strip()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scene JSON not found: {path}")
        out[name] = Scene.from_json(path)
    if not out:
        raise ValueError("No valid models parsed from --compare_models.")
    return out

def main(args: argparse.Namespace) -> None:
    ref = Point2D(args.refx, args.refy)

    # --- Mode A: Multi-model comparison (2×5 layout)
    if args.compare_models:
        if not args.out_dir:
            raise ValueError("--out_dir is required for --compare_models.")
        models = _parse_models_arg(args.compare_models)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

        if args.person_id is not None:
            # One figure for a specific person
            out_path = str(Path(args.out_dir) / f"pid_{args.person_id:03d}.png")
            Visualizer.plot_force_comparison_2x5(
                models=models,
                person_id=args.person_id,
                ref_point=ref,
                out_path=out_path,
                show_components=not args.no_components,
                dpi=args.dpi,
            )
        else:
            # Batch: one PNG per person present in ALL models (ID intersection)
            Visualizer.plot_force_comparison_batch_2x5(
                models=models,
                person_id=None,
                ref_point=ref,
                out_dir=args.out_dir,
                show_components=not args.no_components,
                dpi=args.dpi,
            )
        return

    # --- Mode B: Single-scene panels (2×5 layout, treated as 1-model overlay)
    if not args.scene_json:
        raise ValueError("Provide either --compare_models or --scene_json.")
    scene = Scene.from_json(args.scene_json)
    model_name = Path(args.scene_json).stem  # a readable label
    models = {model_name: scene}
    out_dir = args.out_dir or "results/force_panels_2x5"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.person_id is not None:
        out_path = str(Path(out_dir) / f"pid_{args.person_id:03d}.png")
        Visualizer.plot_force_comparison_2x5(
            models=models,
            person_id=args.person_id,
            ref_point=ref,
            out_path=out_path,
            show_components=not args.no_components,
            dpi=args.dpi,
        )
    else:
        Visualizer.plot_force_comparison_batch_2x5(
            models=models,
            person_id=None,
            ref_point=ref,
            out_dir=out_dir,
            show_components=not args.no_components,
            dpi=args.dpi,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Force visualization: 2×5 panels and multi-model comparison.")
    # Common
    parser.add_argument("--refx", type=float, default=0.0, help="Reference X (for angle).")
    parser.add_argument("--refy", type=float, default=0.0, help="Reference Y (for angle).")
    parser.add_argument("--person_id", type=int, default=None, help="Person ID; if omitted, do all (batch).")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for PNGs.")
    parser.add_argument("--no_components", action="store_true", help="Hide decomposed forces (desired/rep_*).")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI (higher = crisper).")

    # Mode A: multi-model comparison
    parser.add_argument("--compare_models", type=str, default=None,
                        help="Comma-separated NAME:path.json pairs, e.g. 'FFN:a.json,FDN:b.json,SFM:c.json'.")

    # Mode B: single-scene panels
    parser.add_argument("--scene_json", type=str, default=None, help="Single Scene JSON with forces.")

    main(parser.parse_args())
