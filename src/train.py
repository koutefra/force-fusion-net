import argparse
import torchmetrics
import os
import datetime
import re
import sys
import torch
import numpy as np
from data.scene_dataset import SceneDataset
from data.julich_caserne_loader import JulichCaserneLoader
from models.direct_net import DirectNet
from models.fusion_net import FusionNet
from models.predictor import Predictor
from evaluation.evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", required=True, type=str, help="Type of model to use (e.g., direct_net).")
parser.add_argument("--batch_size", required=True, type=int, default=64, help="Batch size for training.")
parser.add_argument("--epochs", required=True, type=int, default=5, help="Number of training epochs.")
parser.add_argument("--lr", "--learning_rate", required=False, type=float, default=1e-3, help="Learning rate for the optimizer.")
parser.add_argument("--hidden_dims", required=False, type=lambda x: list(map(int, x.split(','))), default=[512, 512, 256],
                    help="Hidden layer dimensions as a comma-separated list (e.g., 512,512,256).")
parser.add_argument("--pred_steps", required=True, type=int, default=8, help="Number of prediction steps.")
parser.add_argument("--dropout", required=False, type=float, default=0.0, help="Dropout rate for the model.")
parser.add_argument("--dataset_folder", type=str,
                    default="./data/datasets/julich_bottleneck_caserne", help="Path to the dataset folder.")
parser.add_argument("--train_scenes", type=lambda x: x.strip('[]').split(','), 
                    default=['b090', 'b100', 'b110', 'b120', 'b140', 'b180', 'b200', 'b220', 'b250', 'l0', 'l2', 'l4'],
                    help="Training scenes as a comma-separated list (e.g., b090,b100).")
parser.add_argument("--val_scenes", type=lambda x: x.strip('[]').split(','), default=["b160"],
                    help="Validation scenes as a comma-separated list (e.g., b160).")
parser.add_argument("--fdm_win_size", required=False, type=int, default=20, help="Window size for FDM (Feature Descriptor Matrix).")
parser.add_argument("--sampling_step", required=False, type=int, default=1, help="Sampling step size.")
parser.add_argument("--seed", default=21, type=int, help="Random seed.")
parser.add_argument("--device", default="cpu", type=str, help="Device to use (e.g., 'cpu', 'cuda').")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--redirect_output", action="store_true", help="Redirect stdout and stderr to a file if set.")

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
    if args.redirect_output:
        log_file_path = os.path.join(args.logdir, "training_info.txt")
        sys.stdout = open(log_file_path, "w")
        sys.stderr = sys.stdout

    train_paths_names = [(os.path.join(args.dataset_folder, (name + '.txt')), name) for name in args.train_scenes]
    val_paths_names = [(os.path.join(args.dataset_folder, (name + '.txt')), name) for name in args.val_scenes]
    train_dataset = SceneDataset.from_loaders([JulichCaserneLoader(train_paths_names)])
    val_dataset = SceneDataset.from_loaders([JulichCaserneLoader(val_paths_names)])

    train_dataset = train_dataset.approximate_velocities(args.fdm_win_size, "backward")
    val_dataset = val_dataset.approximate_velocities(args.fdm_win_size, "backward")

    metrics = {
        'MAE_all': torchmetrics.MeanAbsoluteError(),
        'MAE_fst': Evaluator.OneStepMAE(0),
        'MAE_trd': Evaluator.OneStepMAE(2),
        'MaxErr_fst': Evaluator.OneStepMaxError(0),
        'MaxErr_trd': Evaluator.OneStepMaxError(2),
    }

    if args.model_type == 'direct_net' or args.model_type == 'fusion_net':
        individual_features_dim = 6
        interaction_features_dim = 5
        obstacle_features_dim = 9
        model_class = DirectNet if args.model_type == 'direct_net' else FusionNet
        model = model_class(
            individual_features_dim,
            interaction_features_dim,
            obstacle_features_dim,
            args.hidden_dims,
            args.dropout
        )
    else:
        raise ValueError(f'No such model {args.model_type}')
    
    model.keras_init(model)
    predictor = Predictor(
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        logdir_path=args.logdir,
    )
    predictor.train(
        train_dataset.scenes,
        val_dataset.scenes,
        pred_steps=args.pred_steps,
        learning_rate=float(args.lr),
        epochs=args.epochs,
        metrics=metrics
    )

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))