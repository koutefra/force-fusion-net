import argparse
import os
import datetime
import re
import numpy as np
import torch
from data.scene_dataset import SceneDataset
from data.loaders.trajnet_loader import TrajnetLoader
from data.torch_dataset import TorchDataset
from models.neural_net_model import NeuralNetModel
from data.feature_extractor import FeatureExtractor
import torchmetrics
import yaml
from pathlib import Path

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
            loaders[name] = TrajnetLoader(path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    scene_collection = SceneDataset(loaders, load_on_demand=args.load_data_on_demand)
    print('Splitting scenes into train/val...')
    scene_collection_train, scene_collection_eval = scene_collection.split(config["val_ratio"])

    train_dataset = TorchDataset(scene_collection_train, device=args.device) 
    eval_dataset = TorchDataset(scene_collection_eval, device=args.device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=train_dataset.prepare_batch)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=config['batch_size'], collate_fn=eval_dataset.prepare_batch)

    model = NeuralNetModel(
        FeatureExtractor.individual_fts_dim, 
        FeatureExtractor.interaction_fts_dim[1], 
        config['interaction_size'], 
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