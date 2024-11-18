import argparse
import os
import yaml
from pathlib import Path
from data.scene_dataset import SceneDataset
from data.loaders.trajnet_loader import TrajnetLoader
from data.loaders.juelich_bneck_loader import JuelichBneckLoader
from data.fdm_calculator import FiniteDifferenceCalculator

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True, type=str, help="Path to the YAML config file.")

def main(args: argparse.Namespace) -> None:
    project_dir = Path(args.config_path).parent.parent.absolute()
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)

    loaders = {}
    for dataset in config['datasets']:
        name = dataset['name']
        path = (project_dir / dataset["input_path"]).resolve()
        dataset_type = dataset["type"]
        compute_fdm = dataset['compute_fdm']

        if compute_fdm:
            win_size = dataset['fdm_settings']['fdm_window_size']
            fdm_calculator = FiniteDifferenceCalculator(win_size=win_size)

        if dataset_type == "trajnet++":
            loaders[name] = TrajnetLoader(path, name, fdm_calculator)
        elif dataset_type == "juelich_bneck":
            loaders[name] = JuelichBneckLoader(path, name, fdm_calculator)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    scene_dataset = SceneDataset(loaders)

    features = scene_dataset.get_features()

    scene_dataset.save_features(features, os.path(config['output_path']), config['save_format'])

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))