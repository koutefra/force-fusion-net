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
from entities.features import IndividualFeatures, InteractionFeatures, ObstacleFeatures
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True, type=str, help="Path to the YAML config file.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device to use (e.g., 'cpu', 'cuda').")

def main(args: argparse.Namespace) -> None:
    project_dir = Path(args.config_path).parent.parent.absolute()
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    dataset_conf = config['datasets'][0]
    name = dataset_conf['name']
    path = (project_dir / dataset_conf["path"]).resolve()
    format = dataset_conf["format"]

    if format == "ndjson":
        features = SceneDataset.load_features_from_ndjson(path)
    else:
        raise ValueError('No supported')

    serialized_features = [
        f for scenes_f in features.values() for scene_f in scenes_f.values() for f in scene_f.to_list()
    ]
    random.shuffle(serialized_features)

    val_size = int(len(serialized_features) * config['val_ratio'])
    train_features = serialized_features[val_size:]
    val_features = serialized_features[:val_size]

    train_dataset = TorchSceneDataset(train_features, device=args.device, dtype=torch.float32) 
    eval_dataset = TorchSceneDataset(val_features, device=args.device, dtype=torch.float32) 

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