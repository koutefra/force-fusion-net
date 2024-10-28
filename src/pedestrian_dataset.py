#!/usr/bin/env python3
from typing import Any, Callable, TypedDict, Tuple, List, Dict
import json
import torch
import argparse
import random
from tqdm import tqdm
import os
import pickle
from vector2d import Position

class PedestrianDataset:

    PositionId = Tuple[int, int]  # (frame_id, person_id)

    class Scene(TypedDict):
        id: int
        p: int
        s: int
        e: int
        positions: Dict["PedestrianDataset.PositionId", Position]
        start_frames: Dict[int, int]  # person_id -> frame_id
        end_frames: Dict[int, int]  # person_id -> frame_id
        frame_ids: List[int]
        person_ids: List[int]
        min_position: Position
        max_position: Position
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

    def __init__(self, path: str, train_test_split: float, cache: bool):
        cache_path = None
        if cache:
            # Insert "_preprocessed_cache" before ".ndjson"
            base, ext = os.path.splitext(path)
            cache_path = f"{base}_preprocessed_cache{ext}.pkl"

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}...")
            with open(cache_path, 'rb') as cache_file:
                cached_data = pickle.load(cache_file)
                self.train_scenes, self.test_scenes = cached_data['train'], cached_data['test']
        else:
            scenes = self._load_ndjson_data(path)

            self.train_scenes, self.test_scenes = self._split_data(scenes, train_test_split)

            if cache_path:
                print(f"Caching dataset to {cache_path}...")
                with open(cache_path, 'wb') as cache_file:
                    pickle.dump({'train': self.train_scenes, 'test': self.test_scenes}, cache_file)

        setattr(self, 'train', self.Dataset(self.train_scenes))
        setattr(self, 'test', self.Dataset(self.test_scenes))

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
            total_lines = sum(1 for _ in file)

        with open(path, 'r') as file:
            print('Loading...')
            for line in tqdm(file, total=total_lines, unit="line", unit_scale=True, mininterval=1.0):
                data = json.loads(line)
                # scenes go first in trajnet++ format
                if 'scene' in data:
                    scene = data['scene']
                    scene_with_track = {
                        **scene,
                        "positions": {},
                        "start_frames": {},
                        "end_frames": {},
                        "frame_ids": [],
                        "person_ids": [],
                        "min_position": Position(x=float("inf"), y=float("inf")),
                        "max_position": Position(x=float("-inf"), y=float("-inf"))
                    }
                    scenes.append(scene_with_track)
                elif 'track' in data:
                    track = data['track']
                    track_f = track['f']
                    track_p = track['p']
                    for scene in scenes:
                        scene_s, scene_e = scene['s'], scene['e']
                        if scene_s <= track_f <= scene_e:
                            pos = Position(x=track['x'], y=track['y'])
                            pos_id = (track_f, track_p)
                            scene['positions'][pos_id] = pos
                            scene['min_position'] = Position(x=min(scene['min_position'].x, pos.x), 
                                                             y=min(scene['min_position'].y, pos.y))
                            scene['max_position'] = Position(x=max(scene['max_position'].x, pos.x), 
                                                             y=max(scene['max_position'].y, pos.y))

        for scene in scenes:
            scene['positions'] = dict(sorted(scene['positions'].items(), key=lambda item: (item[0][0], item[0][1])))

            scene['frame_ids'] = sorted(set(frame for frame, _ in scene['positions'].keys()))
            scene['person_ids'] = sorted(set(ped_id for _, ped_id in scene['positions'].keys()))

            for p in scene['person_ids']:
                frames_for_pedestrian = [frame for frame, ped_id in scene['positions'].keys() if ped_id == p]

                min_frame_id = min(frames_for_pedestrian)
                scene['start_frames'][p] = min_frame_id

                max_frame_id = max(frames_for_pedestrian)
                scene['end_frames'][p] = max_frame_id

        return scenes

def main(path: str, train_test_split: float, cache: bool):
    dataset = PedestrianDataset(path, train_test_split, cache)

    train_scenes = dataset.train._scenes
    selected_scenes = random.sample(train_scenes, 3)

    for scene in selected_scenes:
        print(f"Scene ID: {scene['id']}")
        print(f"Person ID: {scene['p']}")
        print(f"Starting frame: {scene['s']}")
        print(f"Ending frame: {scene['e']}")
        print(f"Min. position: {scene['min_position']}")
        print(f"Max. position: {scene['max_position']}")
        print(f"FPS: {scene['fps']}")
        print(f"Tag: {scene['tag']}")
        
        print("Start frames for each person:")
        for per_id, start_frame in scene['start_frames'].items():
            print(f"Person {per_id}: Start frame {start_frame}")
        
        print("End frames for each pedestrian:")
        for per_id, end_frame in scene['end_frames'].items():
            print(f"Person {per_id}: End frame {end_frame}")
        
        print(f"Frame ids: {scene['frame_ids']}")
        print(f"Person IDs: {scene['person_ids']}")

        print("Positions (records):")
        for pos_id, pos in scene['positions'].items():
            frame_id, person_id = pos_id
            print(f"Frame {frame_id}, Person {person_id}: Position (x={pos.x}, y={pos.y})")

        from visualization import Visualizer
        vis = Visualizer()
        vis.visualize(scene)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pedestrian Dataset Module.")
    parser.add_argument("--path", type=str, 
                        default="./data_trajnet++/test/synth_data/orca_synth.ndjson", 
                        help="Path to a ndjson dataset following the trajnet++ format.")
    parser.add_argument("--cache", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use cache if available (True/False).")
    parser.add_argument("--tr_te_split", type=float, default=0.8, help="Train-test split ratio.") 
    parser.add_argument("--seed", type=int, default=31, help="Random seed.") 
    
    args = parser.parse_args()
    random.seed(args.seed)

    main(args.path, args.tr_te_split, args.cache)