import argparse
import random
import os
import datetime
import re
import numpy as np
import torch
from data.scene_dataset import SceneDataset
from data.loaders.juelich_bneck_loader import JuelichBneckLoader
from models.neural_net_model import NeuralNetModel
from models.neural_net_predictor import NeuralNetPredictor
from models.social_force_model import SocialForceModel
from models.social_force_predictor import SocialForcePredictor
from data.fdm_calculator import FiniteDifferenceCalculator
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

    if config['dataset'] == 'juelich':
        folder_path = (project_dir / config["data_folder_path"]).resolve()
        train_paths_names = [((folder_path / (name + '.txt')).resolve(), name) for name in config['train_scenes']]
        val_paths_names = [((folder_path / (name + '.txt')).resolve(), name) for name in config['val_scenes']]
        fdm_calculator = FiniteDifferenceCalculator(config['fdm_window_size'])
        train_dataset = SceneDataset([JuelichBneckLoader(
            train_paths_names, 
            config['sampling_step'], 
            compute_accelerations=False, 
            fdm_calculator=fdm_calculator)
        ])
        val_dataset = SceneDataset([JuelichBneckLoader(
            val_paths_names, 
            config['sampling_step'], 
            compute_accelerations=False, 
            fdm_calculator=fdm_calculator)
        ])
    else:
        raise ValueError('Dataset ' + config['dataset'] + ' not supported.')

    if config['model_type'] == 'neural_net':
        individual_features_dim = 6
        interaction_features_dim = 5
        obstacle_features_dim = 9
        model = NeuralNetModel(
            individual_features_dim,
            interaction_features_dim,
            obstacle_features_dim,
            config['hidden_dims']
        )
        predictor = NeuralNetPredictor(
            model=model,
            device=args.device,
            batch_size=config['batch_size'],
            logdir_path=args.logdir
        )
        predictor.train(
            train_dataset.scenes,
            val_dataset.scenes,
            prediction_steps=config['prediction_steps'],
            learning_rate=float(config['learning_rate']),
            epochs=config['epochs'],
            save_path=os.path.join(args.logdir, 'neural_net_weights.pth')
        )
    elif config['model_type'] == 'social_force':
        model = SocialForceModel(config['dataset_fps'])
        # predictor = SocialForcePredictor(
        #     model,
        #     logdir_path=args.logdir
        # )
        # predictor.train(
        #     data=serialized_features,
        #     param_grid=SocialForcePredictor.param_ranges_to_param_grid(config['param_ranges']),
        #     save_path=os.path.join(args.logdir, 'social_force_best_grid.json')
        # )
    else:
        raise ValueError(f'No such model {args.model_type}')

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))