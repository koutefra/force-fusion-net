import json
import argparse
import os
import datetime
import re
import numpy as np
import logging
from tqdm import tqdm
from data.scene_dataset import SceneCollection
from data.loaders.trajnet_loader import TrajnetLoader
from entities.frame_object import PersonInFrame
from models.social_force_model import SocialForceModel
from entities.vector2d import Acceleration
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", required=True, type=str, help="The dataset path.")
parser.add_argument("--dataset_type", default='trajnet++', type=str, help="The dataset type.")
parser.add_argument("--metric", default='mse', type=str, choices=["mse", "mae"], help="Either mse or mae.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")

# TO DO: add choice of exhaustive or random grid search

param_grid = [
    {"A": a, "B": b, "tau": tau, "radius": radius, "desired_speed": desired_speed}
    for a in [1.5, 2.0, 2.5]
    for b in [0.3, 0.5]
    for tau in [0.8, 1.0, 1.2]
    for radius in [1.2, 1.5, 1.8]
    for desired_speed in [1.5, 2.0, 2.5]
]

def compute_metric(pred_acc: Acceleration, true_acc: Acceleration, metric_type: str ="mse"):
    errors = [
        ((pred.x - true.x) ** 2 + (pred.y - true.y) ** 2) ** 0.5
        for pred, true in zip(pred_acc, true_acc)
    ]
    if metric_type == "mse":
        return np.mean([e ** 2 for e in errors])
    elif metric_type == "mae":
        return np.mean(errors)
    else:
        raise ValueError("Unknown metric type")

def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)

    # create logdir
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%y-%m-%d_%h%m%s"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.logdir, 'logs.txt'))

    # TO DO: writers.

    # load data
    if args.dataset_type == "trajnet++":
        loader = TrajnetLoader(args.dataset_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    scene_collection = SceneCollection([loader])

    best_metric = float('inf')
    best_grid = None

    for grid in tqdm(param_grid, desc="Computing the best grid..."):
        model = SocialForceModel(**grid)
        metric_values = []

        for scene in scene_collection.scenes.values():
            pred_acc = model.predict_scene(scene)
            true_acc = [
                obj.acceleration
                for frame in scene.frames for obj in frame.frame_objects
                if isinstance(obj, PersonInFrame) and scene.focus_person_id == obj.id
            ]

            metric = compute_metric(pred_acc, true_acc, metric_type=args.metric)
            metric_values.append(metric)

        avg_metric = np.mean(metric_values)

        if avg_metric < best_metric:
            best_metric = avg_metric
            best_grid = grid

    best_grid_path = os.path.join(args.logdir, 'best_params.json')
    with open(best_grid_path, 'w') as f:
        json.dump(best_grid, f, indent=4)
    print(f"Best parameter configuration saved at {best_grid_path}")
    print(f"Best metric ({args.metric}): {best_metric:.4f}")
    print(f"Best grid: {best_grid}")

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))