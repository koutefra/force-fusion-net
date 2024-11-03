from typing import List, Optional, Union, Tuple, Dict
from collections import defaultdict
import json
from dataclasses import dataclass
from core.scene import Scene
from core.vector2d import Point2D
from data.base_loader import BaseLoader

class TrajnetLoader(BaseLoader):
    @dataclass
    class TrajnetTrack:
        f: int  # frame id
        p: int  # person id
        x: float
        y: float

    @dataclass
    class TrajnetScene:
        id: int
        p: int  # primary person id
        s: int  # start frame
        e: int  # end frame
        fps: float
        tag: Optional[Union[int, List[int]]]

    def _parse(self, path: str) -> Tuple[List["TrajnetLoader.TrajnetScene"], List["TrajnetLoader.TrajnetTrack"]]:
        scenes: List[TrajnetLoader.TrajnetScene] = []
        tracks: List[TrajnetLoader.TrajnetTrack] = []

        with open(path, 'r') as file:
            for line in file:
                data = json.loads(line)
                
                if 'scene' in data:
                    scene_data = data['scene']
                    scenes.append(TrajnetLoader.TrajnetScene(
                        id=scene_data['id'],
                        p=scene_data['p'],
                        s=scene_data['s'],
                        e=scene_data['e'],
                        fps=scene_data['fps'],
                        tag=scene_data.get('tag')  # `tag` may be optional
                    ))
                
                elif 'track' in data:
                    track_data = data['track']
                    tracks.append(TrajnetLoader.TrajnetTrack(
                        f=track_data['f'],
                        p=track_data['p'],
                        x=track_data['x'],
                        y=track_data['y']
                    ))

        return scenes, tracks

    def load_scenes(self, path: str, dataset_name: str) -> Dict[int, Scene]:
        scenes, tracks = self._parse(path)

        # map frame IDs to the scenes that include those frames
        scene_lookup = defaultdict(list)
        for scene in scenes:
            for frame in range(scene.s, scene.e + 1):
                scene_lookup[frame].append(scene)

        # initialize a dictionary to collect data in the raw_data format
        scenes_data = {}
        for scene in scenes:
            trajectories = defaultdict(dict)  # {frame_id: {person_id: Point3}}
            obstacles = defaultdict(list)  # {frame_id: [Point1, Point2, ...]}
            focus_person_goals = {}  # {person_id: Point2D}

            # Initialize each scene's data structure without reassigning it
            scenes_data[scene.id] = {
                "id": scene.id,
                "focus_person_ids": [scene.p],
                "focus_person_goals": focus_person_goals,
                "fps": scene.fps,
                "trajectories": trajectories,
                "obstacles": obstacles,
                "tag": scene.tag if isinstance(scene.tag, list) else [scene.tag],
                "dataset": f"trajnet++/{dataset_name}",
            }

        # populate trajectories and focus_person_goals
        for track in tracks:
            for scene in scene_lookup[track.f]:
                scene_data = scenes_data[scene.id]
                if track.f not in scene_data["trajectories"]:
                    scene_data["trajectories"][track.f] = {}
                    scene_data["obstacles"][track.f] = []
                scene_data["trajectories"][track.f][track.p] = Point2D(track.x, track.y)


        # populate the goal destinations for focus persons
        for scene_data in scenes_data.values():
            for person_id in scene_data["focus_person_ids"]:
                person_frames = sorted(
                    [(frame_id, persons[person_id]) for frame_id, persons in scene_data["trajectories"].items() 
                     if person_id in persons],
                    key=lambda x: x[0]
                )
                
                # if the person has frames in the trajectory, set the last observed position as the goal
                if person_frames:
                    last_frame_id, last_position = person_frames[-1]
                    scene_data["focus_person_goals"][person_id] = last_position
            
        # create Scene instances using from_raw_data
        final_scenes = {
            scene_data['id']: Scene.from_raw_data(scene_data)
            for scene_data in scenes_data.values()
        }
        return final_scenes