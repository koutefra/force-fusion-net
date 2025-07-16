import argparse
import numpy as np
import torch
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any
from data.scene_dataset import SceneDataset
from data.julich_caserne_loader import JulichCaserneLoader
from models.direct_net import DirectNet
from models.fusion_net import FusionNet
from models.social_force import SocialForce
from models.predictor import Predictor
from evaluation.evaluator import Evaluator
from evaluation.visualizer import Visualizer
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description="Comprehensive evaluation of pedestrian models")
parser.add_argument("--dataset_path", required=True, type=str, help="Path to dataset")
parser.add_argument("--test_scenes", type=lambda x: x.strip('[]').split(','), 
                    default=['b090', 'b100', 'b110', 'b120', 'b140', 'b160', 'b180', 'b200', 'b220', 'b250', 'l0', 'l2', 'l4'],
                    help="Test scenes as comma-separated list")
parser.add_argument("--model_paths", type=str, nargs='+', required=True, 
                    help="Paths to trained model weights")
parser.add_argument("--model_types", type=str, nargs='+', required=True,
                    help="Model types corresponding to model paths")
parser.add_argument("--model_names", type=str, nargs='+', 
                    help="Custom names for models (optional)")
parser.add_argument("--baseline_comparisons", type=str, nargs='*', 
                    default=['social_force', 'gt'],
                    help="Baseline methods to compare against")
parser.add_argument("--fdm_win_size", default=20, type=int, help="FDM window size")
parser.add_argument("--simulation_steps", default=500, type=int, help="Number of simulation steps")
parser.add_argument("--goal_radius", default=0.4, type=float, help="Goal radius for simulations")
parser.add_argument("--output_dir", default="evaluation_results", type=str, help="Output directory")
parser.add_argument("--device", default="cpu", type=str, help="Device to use")
parser.add_argument("--seed", default=21, type=int, help="Random seed")



def main(args):
    evaluator = ComprehensiveEvaluator(args.output_dir)
    results = evaluator.run_evaluation(args)
    return results

if __name__ == "__main__":
    main(parser.parse_args([] if "__file__" not in globals() else None))