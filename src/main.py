from data.dataset import PedestrianDataset
from models.social_force_model import SocialForcePredictor
from ground_truth_predictor import GroundTruthPredictor
from visualization import Visualizer
import argparse
import random

def main(dataset: PedestrianDataset):
    train_scenes = dataset.train._scenes
    selected_scenes = random.sample(train_scenes, 3)
    ground_truth = GroundTruthPredictor()
    social_force = SocialForcePredictor()

    for scene in selected_scenes:
        gt_prediction = ground_truth.predict(scene)
        vis = Visualizer()
        vis.visualize(scene, time_scale=2.0, name='Ground Truths', prediction=gt_prediction)

        social_force_prediction = social_force.predict(scene)
        vis = Visualizer()
        vis.visualize(scene, time_scale=2.0, name='Social Force', prediction=social_force_prediction)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Dataset Module.")
    parser.add_argument("--path", type=str, 
                        default="./data_trajnet++/train/real_data/lcas.ndjson", 
                        # default="./data_trajnet++/train/real_data/biwi_hotel.ndjson", 
                        # default="./data_trajnet++/test/synth_data/orca_synth.ndjson", 
                        help="Path to a ndjson dataset following the trajnet++ format.")
    parser.add_argument("--cache", type=lambda x: (str(x).lower() == 'true'), default=True, help="Cache dataset after processing (True/False).")
    parser.add_argument("--tr_te_split", type=float, default=0.8, help="Train-test split ratio.") 
    parser.add_argument("--seed", type=int, default=31, help="Random seed.") 
    
    args = parser.parse_args()
    random.seed(args.seed)

    dataset = PedestrianDataset(args.path, args.tr_te_split, args.cache)

    main(dataset)
