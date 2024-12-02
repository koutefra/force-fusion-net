import argparse
import random
import os
import datetime
import re
import numpy as np
import torch
import torchmetrics
from data.scene_dataset import SceneDataset
from data.torch_dataset import TorchSceneDataset
from models.neural_net_model import NeuralNetModel
from models.neural_net_predictor import NeuralNetPredictor
from models.social_force_model import SocialForceModel
from models.social_force_predictor import SocialForcePredictor
from entities.features import IndividualFeatures, InteractionFeatures, ObstacleFeatures
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Path to the YAML config file.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device to use (e.g., 'cpu', 'cuda').")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def main(args: argparse.Namespace) -> None:
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    project_dir = Path(args.config).parent.parent.absolute()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    path = (project_dir / config["dataset_path"]).resolve()
    format = config["dataset_format"]

    if format == "ndjson":
        features = SceneDataset.load_features_from_ndjson(
            path, 
            n_samples=config['n_samples'] if 'n_samples' in config else None
        )
    elif format == "pickle":
        features = SceneDataset.load_features_from_pickle(path)
    else:
        raise ValueError(f'Input data format not supported')

    serialized_features = [
        f for scenes_f in features.values() for scene_f in scenes_f.values() for f in scene_f.to_list()
    ]
    random.shuffle(serialized_features)

    if config['model_type'] == 'neural_net':
        model = NeuralNetModel(
            IndividualFeatures.dim(), 
            InteractionFeatures.dim(), 
            ObstacleFeatures.dim(),
            config['hidden_dim']
        )
        predictor = NeuralNetPredictor(
            model=model,
            batch_size=config['batch_size'],
            logdir_path=args.logdir,
            device=args.device
        )

        # split to train/val
        val_size = int(len(serialized_features) * config['val_ratio'])
        train_features = serialized_features[val_size:]
        val_features = serialized_features[:val_size]

        predictor.train(
            train_features,
            val_features,
            learning_rate=float(config['learning_rate']),
            epochs=config['epochs'],
            save_path=os.path.join(args.logdir, 'neural_net_weights.pth')
        )
    elif config['model_type'] == 'social_force':
        model = SocialForceModel(config['dataset_fps'])
        predictor = SocialForcePredictor(
            model,
            logdir_path=args.logdir
        )
        predictor.train(
            data=serialized_features,
            param_grid=SocialForcePredictor.param_ranges_to_param_grid(config['param_ranges']),
            save_path=os.path.join(args.logdir, 'social_force_best_grid.json')
        )
    else:
        raise ValueError(f'No such model {args.model_type}')

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))