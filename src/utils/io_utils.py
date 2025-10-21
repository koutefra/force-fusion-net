from pathlib import Path
import numpy as np
import random
import torch
from data.scene_dataset import SceneDataset
from data.julich_caserne_loader import JulichCaserneLoader
from models.predictor import Predictor
from models.direct_net import DirectNet
from models.fusion_net import FusionNet
from models.social_force import SocialForce
from models.social_force_b160 import SocialForceB160
from entities.scene import Scene


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def load_scene(dataset_folder: str | Path, scene_file: str, scene_name: str | None, fdm_win_size: int = 20):
    """
    Load a scene JSON into a SceneDataset and approximate velocities.

    Args:
        dataset_folder: Path to dataset directory.
        scene_file: JSON file name (e.g., 'b160.json').
        scene_name: Optional scene name; defaults to stem of scene_file.
        fdm_win_size: Finite difference method window size.

    Returns:
        (scene, dataset)
    """
    dataset_folder = Path(dataset_folder)
    scene_path = dataset_folder / scene_file

    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    if not scene_name:
        scene_name = scene_path.stem  # filename without extension

    dataset = SceneDataset.from_loaders([
        JulichCaserneLoader([(str(scene_path), scene_name)])
    ])
    dataset = dataset.approximate_velocities(fdm_win_size, "backward")
    scene = next(iter(dataset.scenes.values()))

    return scene, dataset


def load_model(model_folder: str | Path, model_file: str, model_type: str, device: str = "cpu"):
    """
    Load a pedestrian prediction model by type and wrap it in a Predictor.

    Args:
        model_folder: Directory where model weights are stored.
        model_file: Weight file name.
        model_type: One of ['fusion_net', 'direct_net', 'social_force', 'social_force_b160'].
        device: 'cpu' or 'cuda'.

    Returns:
        Predictor instance wrapping the loaded model.
    """
    model_folder = Path(model_folder)
    model_path = model_folder / model_file

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if model_type == "fusion_net":
        model = FusionNet.from_weight_file(str(model_path))
    elif model_type == "direct_net":
        model = DirectNet.from_weight_file(str(model_path))
    elif model_type == "social_force":
        model = SocialForce.from_weight_file(str(model_path))
    elif model_type == "social_force_b160":
        model = SocialForceB160.from_weight_file(str(model_path))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return Predictor(model, device=device)

def parse_models_arg(models_arg: str) -> dict[str, Scene]:
    """
    Parse a comma-separated list like:
        'FusionNet:path1.json,DirectNet:path2.json'
    into:
        {'FusionNet': Scene(...), 'DirectNet': Scene(...)}
    """
    if not models_arg:
        raise ValueError("Missing --models or --compare_models argument.")

    out: dict[str, Scene] = {}
    parts = [p.strip() for p in models_arg.split(",") if p.strip()]

    for entry in parts:
        if ":" not in entry:
            raise ValueError(f"Malformed entry: '{entry}' (expected NAME:path.json)")
        name, path = entry.split(":", 1)
        name, path = name.strip(), path.strip()

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scene JSON not found: {p}")

        out[name] = Scene.from_json(str(p))

    if not out:
        raise ValueError("No valid models parsed from argument string.")

    return out