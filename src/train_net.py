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
from data.scene_collection import PedestrianDataset
from data.trajnet_loader import TrajnetLoader
from data.feature_extractor import SceneProcessor
from data.torch_dataset import TorchDataset
from core.scene_datapoint import SceneDatapoint
from typing import Dict, Tuple, List
from models.neural_net_model import NeuralNetModel
from torch.nn.utils.rnn import pad_sequence
import torchmetrics

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
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

    dataset = PedestrianDataset(loader, args.dataset_path, args.dataset_name)
    train_dataset, eval_dataset = dataset.split(args.val_ratio)
    scene_processor = SceneProcessor(include_focus_persons_only=True)
    train_dataset = TorchDataset(train_dataset.get_scenes(), scene_processor) 
    eval_dataset = TorchDataset(eval_dataset.get_scenes(), scene_processor)

    def prepare_example(datapoint: Dict[str, List[float]], data_id: Dict[str, int]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        person_features = torch.tensor(list(datapoint["person_features"].values()), dtype=torch.float32)

        interaction_features = torch.tensor([list(it_fts.values()) for it_fts in datapoint["interaction_features"]], dtype=torch.float32) 
        if len(interaction_features) == 0:
            interaction_features = torch.zeros(1, SceneProcessor.INTERACTION_FEATURES_DIM[1])
            
        obstacle_features = torch.tensor([list(ob_fts.values()) for ob_fts in datapoint["obstacle_features"]], dtype=torch.float32)
        if len(obstacle_features) == 0:
            obstacle_features = torch.zeros(1, SceneProcessor.OBSTACLE_FEATURES_DIM[1])

        label = torch.tensor(list(datapoint["label"].values()), dtype=torch.float32)
        return (person_features, interaction_features, obstacle_features), label, data_id

    train_dataset = train_dataset.transform(prepare_example)
    eval_dataset = eval_dataset.transform(prepare_example)

    def prepare_batch(data: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, Dict[str, int]]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, Dict[str, int]], torch.Tensor]:
        input, output, metadata = zip(*data)
        person_features, interaction_features, obstacle_features = zip(*input)
        person_features_stack = torch.stack(person_features).float()
        interaction_features_stack = pad_sequence(interaction_features, batch_first=True).float()
        obstacles_features_stack = pad_sequence(obstacle_features, batch_first=True).float()
        inputs = (person_features_stack, interaction_features_stack)
        outputs = torch.stack(output).float()
        return inputs, outputs, metadata

    train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=prepare_batch)
    eval = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=prepare_batch)

    model = NeuralNetModel(SceneProcessor.PERSON_FEATURES_DIM, SceneProcessor.INTERACTION_FEATURES_DIM[1], 
                           args.interaction_size, args.hidden_sizes, SceneProcessor.LABEL_DIM)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        device=args.device,
        logdir=args.logdir,
        metrics={'MAE': torchmetrics.MeanAbsoluteError()},
        loss=torch.nn.MSELoss()
    )

    logs = model.fit(train, dev=eval, epochs=args.epochs, callbacks=[])
    model.save_weights(os.path.join(args.logdir, 'weights.pth'))

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))