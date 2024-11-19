import argparse
import os
import yaml
from pathlib import Path
from data.scene_dataset import SceneDataset
from data.loaders.trajnet_loader import TrajnetLoader
from data.loaders.juelich_bneck_loader import JuelichBneckLoader
from data.fdm_calculator import FiniteDifferenceCalculator

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, type=str, help="Path to the YAML config file.")

def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    for dataset in config['datasets']:
        if 'juelich_bneck' in dataset:
            dataset_config = dataset['juelich_bneck']
            compute_fdm = dataset_config['compute_fdm']
            input_folder = dataset_config['input_folder']
            sampling_step = int(dataset_config['sampling_step'])
            names = dataset_config['names']
            paths_and_names = [(os.path.join(input_folder, f"{name}.txt"), name) for name in names]
            if compute_fdm:
                window_size = dataset_config['fdm_settings']['fdm_window_size']
                fdm_calculator = FiniteDifferenceCalculator(window_size)

            for path, name in paths_and_names:
                loader = JuelichBneckLoader([(path, name)], sampling_step, fdm_calculator)
                scene_dataset = SceneDataset({'juelich_bneck': loader})
                features = scene_dataset.get_labeled_features()
                filepath = os.path.join(config['output_path'], f'juelich_bneck_labeled_sstep{sampling_step}')
                scene_dataset.save_features_as_ndjson(features, filepath, writing_mode="a")
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))