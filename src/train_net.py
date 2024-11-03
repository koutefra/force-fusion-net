#!/usr/bin/env python3
import argparse
import os
import datetime
import re
import numpy as np
from random import randrange
import torch
from torch import nn
import torch.nn.functional as F
from data.pedestrian_dataset import PedestrianDataset
from data.trajnet_loader import TrajnetLoader
from data.basic_scene_processor import SceneFeatureExtractor
from data.torch_dataset import TorchPedestrianDataset
from typing import Dict, Tuple, List
from models.neural_net_model import NeuralNetModel
from torch.nn.utils.rnn import pad_sequence
import torchmetrics

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--model_path", type=str, help="Path where to store the model's weights.")
parser.add_argument("--dataset_type", default='trajnet++', type=str, help="The dataset type.")
parser.add_argument("--dataset_name", default='orca_synth_train', type=str, help="The dataset name.")
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

    # load data
    if args.dataset_type == "trajnet++":
        loader = TrajnetLoader()
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    feature_extractor = SceneFeatureExtractor(True)
    pedestrian_dataset = PedestrianDataset(loader, feature_extractor, args.dataset_path, args.dataset_name)
    train_dataset, eval_dataset = pedestrian_dataset.split(args.val_ratio)
    train_dataset, eval_dataset = TorchPedestrianDataset(train_dataset), TorchPedestrianDataset(eval_dataset)

    ex_data = train_dataset[0]
    ex_person_features, ex_interaction_features, ex_obstacle_features, ex_label, ex_metadata = ex_data
    person_features_dim = ex_person_features.shape[0]
    interaction_features_dim = ex_interaction_features.shape[1]
    label_dim = ex_label.shape[0]
    # print(ex_person_features.shape, ex_interaction_features.shape, ex_label.shape)

    def prepare_batch(data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        person_features_ts, interaction_features_ts, obstacle_features_ts, label_ts, metadata = zip(*data)
        person_features_stack = torch.stack(person_features_ts).float() 
        interaction_features_stack = pad_sequence(interaction_features_ts, batch_first=True).float()
        inputs = (person_features_stack, interaction_features_stack)
        outputs = torch.stack(label_ts).float()
        # print(inputs[0].shape, inputs[1].shape, outputs.shape)
        return inputs, outputs

    train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=prepare_batch)
    eval = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=prepare_batch)

    model = NeuralNetModel(person_features_dim, interaction_features_dim, args.interaction_size, 
                           args.hidden_sizes, label_dim)

    class MyMeanAbsoluteError(torchmetrics.Metric):
        def __init__(self):
            super().__init__()
            # Add two state variables for sum of absolute errors and count
            self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            # Update the sum of absolute errors and count
            self.sum_abs_error += torch.sum(torch.abs(preds - target))
            self.count += target.numel()  # Count the number of elements in the target

        def compute(self):
            # Compute the mean absolute error
            return self.sum_abs_error / self.count

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        device=args.device,
        logdir=args.logdir,
        metrics={'MAE': torchmetrics.MeanAbsoluteError(), 'MyMAE': MyMeanAbsoluteError()},
        loss=torch.nn.MSELoss()
    )

    logs = model.fit(train, dev=eval, epochs=args.epochs, callbacks=[])
    # if args.model_path:
        # model.save_weights(args.model_path)

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))