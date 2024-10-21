#!/usr/bin/env python3
from typing import Any, Callable, TypedDict, Optional, Tuple, List, Dict
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random


class PedestrianDataset:

    class Track(TypedDict):
        f: int
        p: int
        x: float
        y: float

    class Scene(TypedDict):
        id: int
        p: int
        s: int
        e: int
        tracks: Dict[List["PedestrianDataset.Track"]]
        s_track_id: int
        e_track_id: int
        fps: float
        tag: int

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, scenes: List["PedestrianDataset.Scene"]) -> None:
            self._scenes = scenes 
            self._size = len(self._scenes)

        def __len__(self) -> int:
            return self._size

        def __getitem__(self, index: int) -> "PedestrianDataset.Scene":
            return self._scenes[index]

        def transform(self, transform):
            return PedestrianDataset.TransformedDataset(self, transform)

    class TransformedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset: torch.utils.data.Dataset, transform: Callable[..., Any]) -> None:
            self._dataset = dataset
            self._transform = transform

        def __len__(self) -> int:
            return len(self._dataset)

        def __getitem__(self, index: int) -> Any:
            item = self._dataset[index]
            return self._transform(*item) if isinstance(item, tuple) else self._transform(item)

        def transform(self, transform: Callable[..., Any]) -> "PedestrianDataset.TransformedDataset":
            return PedestrianDataset.TransformedDataset(self, transform)

    def __init__(self, path: str, train_test_split: float):
        scenes = self._load_ndjson_data(path)

        train_scenes, test_scenes = self._split_data(scenes, train_test_split)
        
        setattr(self, 'train', self.Dataset(train_scenes))
        setattr(self, 'test', self.Dataset(test_scenes))

    train: Dataset
    test: Dataset

    @staticmethod
    def _split_data(data: List[Any], train_test_split: float):
        data_indices = list(range(len(data)))
        random.shuffle(data_indices)
        
        split_index = int(len(data_indices) * train_test_split)
        
        train_data_indices = data_indices[:split_index]
        test_data_indices = data_indices[split_index:]
        
        train_data = [data[i] for i in train_data_indices]
        test_data = [data[i] for i in test_data_indices]
        
        return train_data, test_data

    @staticmethod
    def _load_ndjson_data(path: str) -> List["PedestrianDataset.Scene"]:
        scenes: "PedestrianDataset.Scene" = []
        with open(path, 'r') as file:
            for line in file:
                data = json.loads(line)
                # scenes go first in trajnet++ format
                if 'scene' in data:
                    scene = data['scene']
                    scene_with_track = {
                        **scene,
                        "tracks": [],
                        "s_track_id": -1,
                        "e_track_id": -1
                    }
                    scenes.append(scene_with_track)
                elif 'track' in data:
                    track = data['track']
                    for scene in scenes:
                        if scene['s'] <= track['f'] <= scene['e']:
                            scene['tracks'].append(track)

        for scene in scenes:
            scene['tracks'] = sorted(scene['tracks'], key=lambda x: x['f'])
            p_tracks = [track for track in scene['tracks'] if track['p'] == scene['p']]
            scene['s_track_id'] = p_tracks[0]
            scene['e_track_id'] = p_tracks[-1]

        return scenes



def main(path: str, train_test_split: float):
    dataset = PedestrianDataset(path, train_test_split)

    print(f"scene_id: {dataset.train[0]['id']}")
    print(f"pedestrian_id: {dataset.train[0]['p']}")
    print(f"starting_frame: {dataset.train[0]['s']}")
    print(f"ending_frame: {dataset.train[0]['e']}")
    print(f"fps: {dataset.train[0]['fps']}")
    print(f"tag: {dataset.train[0]['tag']}")
    print(f"s_track_id: {dataset.train[0]['s_track_id']}")
    print(f"e_track_id: {dataset.train[0]['e_track_id']}")
    print(f"tracks:")
    for track in dataset.train[0]['tracks']:
        print(track)

    from visualization import visualize
    visualize(dataset.train[0], res=(800, 600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Dataset Module.")
    parser.add_argument("--path", type=str, 
                        default="/home/koutefra/projects/evacuation/data_trajnet++/train/real_data/biwi_hotel.ndjson", 
                        help="Path to a ndjson dataset following the trajnet++ format.")
    parser.add_argument("--tr_te_split", type=float, default=0.8, help="Train-test split ratio.") 
    parser.add_argument("--seed", type=int, default=21, help="Random seed.") 

    args = parser.parse_args()
    random.seed(args.seed)
    main(args.path, args.tr_te_split)