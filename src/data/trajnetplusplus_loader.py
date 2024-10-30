from typing import List, Optional, Union
from collections import defaultdict
import json
from dataclasses import dataclass
from core.scene import Scene
from data.base_loader import BaseLoader

class TrajnetPlusPlusLoader(BaseLoader):
    @dataclass
    class TrajnetPlusPlusTrack:
        f: int  # frame id
        p: int  # person id
        x: float
        y: float

    @dataclass
    class TrajnetPlusPlusScene:
        id: int
        p: int  # primary person id
        s: int  # start frame
        e: int  # end frame
        fps: float
        tag: Optional[Union[int, List[int]]]

    def __init__(self, dataset_name: str):
        super().__init__(dataset_name)
        self.scenes: List[TrajnetPlusPlusLoader.TrajnetPlusPlusScene] = []
        self.tracks: List[TrajnetPlusPlusLoader.TrajnetPlusPlusTrack] = []
        
    def load(self, path: str) -> None:
        with open(path, 'r') as file:
            for line in file:
                data = json.loads(line)
                
                if 'scene' in data:
                    scene_data = data['scene']
                    self.scenes.append(TrajnetPlusPlusLoader.TrajnetPlusPlusScene(
                        id=scene_data['id'],
                        p=scene_data['p'],
                        s=scene_data['s'],
                        e=scene_data['e'],
                        fps=scene_data['fps'],
                        tag=scene_data.get('tag')  # `tag` may be optional
                    ))
                
                elif 'track' in data:
                    track_data = data['track']
                    self.tracks.append(TrajnetPlusPlusLoader.TrajnetPlusPlusTrack(
                        f=track_data['f'],
                        p=track_data['p'],
                        x=track_data['x'],
                        y=track_data['y']
                    ))

    def preprocess(self) -> List[Scene]:
        # Step 1: Map frame IDs to the scenes that include those frames
        scene_lookup = defaultdict(list)
        for scene in self.scenes:
            for frame in range(scene.s, scene.e + 1):
                scene_lookup[frame].append(scene)

        # Step 2: Initialize a dictionary to collect data in the raw_data format
        scenes_data = {}
        for scene in self.scenes:
            trajectories = []
            obstacles = defaultdict(list)

            scenes_data[scene.id] = {
                "id": scene.id,
                "focus_person_ids": [scene.p],
                "fps": scene.fps,
                "trajectories": trajectories,
                "obstacles": obstacles,
                "tag": scene.tag if isinstance(scene.tag, list) else [scene.tag],
                "dataset": f"trajnet++/{self.dataset_name}",
            }

        # Step 3: Populate trajectories
        for track in self.tracks:
            for scene in scene_lookup[track.f]:
                scene_data = scenes_data[scene.id]
                scene_data["trajectories"].append((track.f, track.p, {"x": track.x, "y": track.y}))

        # Step 4: Create Scene instances using from_raw_data
        final_scenes = [
            Scene.from_raw_data(scene_data)
            for scene_data in scenes_data.values()
        ]

        return final_scenes

def main(path: str, dataset_name: str):
    loader = TrajnetPlusPlusLoader(dataset_name)
    loader.load(path)
    scenes = loader.preprocess()

    for i, scene in enumerate(scenes[:3]):
        print(f"Scene ID: {scene.id}")
        print(f"Primary person IDs: {scene.focus_person_ids}")
        print(f"Starting frame: {scene.sorted_frame_ids[0] if scene.sorted_frame_ids else 'N/A'}")
        print(f"Ending frame: {scene.sorted_frame_ids[-1] if scene.sorted_frame_ids else 'N/A'}")
        print(f"FPS: {scene.fps}")
        print(f"Tag: {scene.tag}")
        
        start_frames, end_frames = scene.person_frame_ranges
        print("Start frames for each person:")
        for per_id, start_frame in start_frames.items(): 
            print(f"Person {per_id}: Start frame {start_frame}")
        
        print("End frames for each pedestrian:")
        for per_id, end_frame in end_frames.items(): 
            print(f"Person {per_id}: End frame {end_frame}")
        
        print("Trajectories:")
        for frame_id, person in scene.trajectories.items():
            for person_id, point in person.items():
                print(f"Frame {frame_id}, Person {person_id}: Point (x={point.x}, y={point.y})")

        print("\n" + "-"*40 + "\n")  # Separator for readability

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Trajnet++ Loader Module.")
    parser.add_argument("--path", type=str, required=True, help="Path to a ndjson dataset following the trajnet++ format.")
    parser.add_argument("--dataset_name", type=str, default='unknown', help="Name of the trajnet++ dataset")
    args = parser.parse_args()
    main(args.path, args.dataset_name)
