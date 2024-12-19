import argparse
import random
import os
import datetime
import re
import torch
import numpy as np
from data.scene_dataset import SceneDataset
from data.julich_caserne_loader import JulichCaserneLoader
from models.direct_net import DirectNet
from models.predictor import Predictor
from models.social_force import SocialForce
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

    if config['dataset'] == 'julich':
        folder_path = (project_dir / config["data_folder_path"]).resolve()
        train_paths_names = [((folder_path / (name + '.txt')).resolve(), name) for name in config['train_scenes']]
        val_paths_names = [((folder_path / (name + '.txt')).resolve(), name) for name in config['val_scenes']]
        train_dataset = SceneDataset.from_loaders([JulichCaserneLoader(train_paths_names)])
        val_dataset = SceneDataset.from_loaders([JulichCaserneLoader(val_paths_names)])
    else:
        raise ValueError('Dataset ' + config['dataset'] + ' not supported.')

    train_dataset = train_dataset.approximate_velocities(config['fdm_window_size'], "backward")
    val_dataset = val_dataset.approximate_velocities(config['fdm_window_size'], "backward")

    if config['model_type'] == 'direct_net':
        individual_features_dim = 6
        interaction_features_dim = 5
        obstacle_features_dim = 9
        model = DirectNet(
            individual_features_dim,
            interaction_features_dim,
            obstacle_features_dim,
            config['hidden_dims'],
            config['dropout']
        )
        DirectNet.keras_init(model)
    elif config['model_type'] == 'social_force':
        model = SocialForce(**config['weights']['params'])
        if config['weights']['keras_random']:
            SocialForce.keras_init(model)
    else:
        raise ValueError(f'No such model {args.model_type}')

    predictor = Predictor(
        model=model,
        device=args.device,
        batch_size=config['batch_size'],
        logdir_path=args.logdir
    )
    predictor.train(
        train_dataset.scenes,
        val_dataset.scenes,
        pred_steps=config['pred_steps'],
        learning_rate=float(config['learning_rate']),
        epochs=config['epochs']
    )

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))