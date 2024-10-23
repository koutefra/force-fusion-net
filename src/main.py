from pedestrian_dataset import PedestrianDataset
from social_force import SocialForcePredictor
from visualization import Visualizer
import argparse
import random

def main(dataset: PedestrianDataset):
    print(f"scene_id: {dataset.train[0]['id']}")
    print(f"pedestrian_id: {dataset.train[0]['p']}")
    print(f"starting_frame: {dataset.train[0]['s']}")
    print(f"ending_frame: {dataset.train[0]['e']}")
    print(f"fps: {dataset.train[0]['fps']}")
    print(f"tag: {dataset.train[0]['tag']}")
    print(f"s_track_id: {dataset.train[0]['s_track_id']}")
    print(f"e_track_id: {dataset.train[0]['e_track_id']}")
    print(f"records:")
    for track in dataset.train[0]['records']:
        print(track)

    from visualization import visualize
    # visualize(dataset.train[0], res=(800, 600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Model.")
    parser.add_argument("--data_path", type=str, 
                        default="./data_trajnet++/train/real_data/biwi_hotel.ndjson", 
                        help="Path to a ndjson dataset following the trajnet++ format.")
    parser.add_argument("--res", type=int, nargs=2, default=[800, 600], help="Resolution as width height.")
    parser.add_argument("--tr_te_split", type=float, default=0.8, help="Train-test split ratio.") 
    parser.add_argument("--seed", type=int, default=31, help="Random seed.") 

    args = parser.parse_args()
    random.seed(args.seed)

    dataset = PedestrianDataset(args.data_path, args.train_test_split)
    visualizer = Visualizer(args.res)
    main(dataset, visualizer)