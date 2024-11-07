import argparse
import os
import datetime
import re
import numpy as np
import torch
from data.scene_collection import SceneCollection
from data.trajnet_loader import TrajnetLoader
from data.torch_dataset import TorchDataset
from models.neural_net_model import NeuralNetModel
from data.feature_extractor import FeatureExtractor
import torchmetrics

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--dataset_type", default='trajnet++', type=str, help="The dataset type.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device.")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--interaction_size", default=512, type=int, help="Number of hidden channels.")
parser.add_argument("--hidden_sizes", default=[1024], type=int, nargs='+', help="List of hidden channel sizes.")
parser.add_argument("--val_ratio", default=0.2, type=float, help="Train/val data ratio.")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create logdir
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    # load data
    if args.dataset_type == "trajnet++":
        loader = TrajnetLoader(args.dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    scene_collection = SceneCollection([loader])
    scene_collection_train, scene_collection_val = scene_collection.split(args.val_ratio)

    train_dataset = TorchDataset(scene_collection_train) 
    val_dataset = TorchDataset(scene_collection_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=TorchDataset.prepare_batch)
    eval_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=TorchDataset.prepare_batch)

    model = NeuralNetModel(FeatureExtractor.individual_fts_dim, FeatureExtractor.interaction_fts_dim[1], 
                           args.interaction_size, args.hidden_sizes, 2)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        device=args.device,
        logdir=args.logdir,
        metrics={'MAE': torchmetrics.MeanAbsoluteError()},
        loss=torch.nn.MSELoss()
    )

    logs = model.fit(train_loader, dev=eval_loader, epochs=args.epochs, callbacks=[])
    model.save_weights(os.path.join(args.logdir, 'weights.pth'))

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))