import argparse
import os
import datetime
import re
import numpy as np
import torch
from data.scene_dataset import SceneDataset
from data.data_processor import EagerProcessor, LazyProcessor
from data.loaders.trajnet_loader import TrajnetLoader
from data.loaders.juelich_bneck_loader import JuelichBneckLoader
from data.torch_dataset import TorchSceneDataset
from models.neural_net_model import NeuralNetModel
from data.feature_extractor import FeatureExtractor
from entities.features import IndividualFeatures, InteractionFeatures, ObstacleFeatures
import torchmetrics
import yaml
from pathlib import Path
from data.fdm_calculator import FiniteDifferenceCalculator

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True, type=str, help="Path to the YAML config file.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device to use (e.g., 'cpu', 'cuda').")
parser.add_argument("--load_data_on_demand", action="store_true", help="Whether to load data on demand.")

def main(args: argparse.Namespace) -> None:
    project_dir = Path(args.config_path).parent.parent.absolute()
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create logdir
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    loaders = {}
    for dataset in config['datasets']:
        name = dataset['name']
        path = (project_dir / dataset["path"]).resolve()
        dataset_type = dataset["type"]

        if dataset_type == "trajnet++":
            fdm_calculator = FiniteDifferenceCalculator(win_size=2)
            loaders[name] = TrajnetLoader(path, name, fdm_calculator)
        elif dataset_type == "juelich_bneck":
            fdm_calculator = FiniteDifferenceCalculator(win_size=20)
            loaders[name] = JuelichBneckLoader(path, name, fdm_calculator)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    if args.load_data_on_demand:
        processor = LazyProcessor(loaders)
    else:
        processor = EagerProcessor(loaders)
    scene_collection = SceneDataset(processor)
    print('Splitting scenes into train/val...')
    scene_collection_train, scene_collection_eval = scene_collection.split(config["val_ratio"])

    feature_extractor = FeatureExtractor(print_progress=False)
    train_dataset = TorchSceneDataset(scene_collection_train, device=args.device, feature_extractor=feature_extractor) 
    eval_dataset = TorchSceneDataset(scene_collection_eval, device=args.device, feature_extractor=feature_extractor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=train_dataset.prepare_batch)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config['batch_size'], collate_fn=eval_dataset.prepare_batch)

    model = NeuralNetModel(
        IndividualFeatures.dim(), 
        InteractionFeatures.dim(), 
        ObstacleFeatures.dim(),
        config['interaction_out_dim'], 
        config['obstacle_out_dim'], 
        config['hidden_sizes'], 
        output_dim=2
    )

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=float(config['learning_rate'])),
        device=args.device,
        logdir=args.logdir,
        metrics={'MAE': torchmetrics.MeanAbsoluteError()},
        loss=torch.nn.MSELoss()
    )

    logs = model.fit(train_loader, dev=eval_loader, epochs=config['epochs'], callbacks=[])
    model.save_weights(os.path.join(args.logdir, 'weights.pth'))

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))