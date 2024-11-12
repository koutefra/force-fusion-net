import json
from data.base_loader import BaseLoader
from entities.raw_scenes_data import RawSceneData

class TrajnetLoader(BaseLoader):
    def load_scene_by_id(self, scene_id: int) -> RawSceneData:
        return self.load_scenes_by_ids({scene_id: True})

    def load_scenes_by_ids(self, scene_ids: set[int]) -> RawSceneData:
        scenes = []
        tracks = []
        with open(self.path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'scene' in data and data['scene']['id'] in scene_ids:
                    scene_data = data['scene']
                    scenes.append(RawSceneData.SceneData(
                        id=scene_data['id'],
                        person_id=scene_data['p'],
                        start_frame_number=scene_data['s'],
                        end_frame_number=scene_data['e'],
                        fps=scene_data['fps'],
                        tag=scene_data.get('tag')
                    ))
                elif 'track' in data:
                    track_data = data['track']
                    for scene in scenes:
                        if scene.start_frame_number <= track_data['f'] <= scene.end_frame_number:
                            tracks.append(RawSceneData.TrackData(
                                frame_number=track_data['f'],
                                person_id=track_data['p'],
                                x=track_data['x'],
                                y=track_data['y']
                            ))
        return RawSceneData(scenes, tracks)
    
    def load_all_scenes(self) -> RawSceneData:
        scenes = []
        tracks = []
        with open(self.path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'scene' in data:
                    scene_data = data['scene']
                    scenes.append(RawSceneData.SceneData(
                        id=scene_data['id'],
                        person_id=scene_data['p'],
                        start_frame_number=scene_data['s'],
                        end_frame_number=scene_data['e'],
                        fps=scene_data['fps'],
                        tag=scene_data.get('tag')
                    ))
                elif 'track' in data:
                    track_data = data['track']
                    tracks.append(RawSceneData.TrackData(
                        frame_number=track_data['f'],
                        person_id=track_data['p'],
                        x=track_data['x'],
                        y=track_data['y']
                    ))
        return RawSceneData(scenes, tracks)