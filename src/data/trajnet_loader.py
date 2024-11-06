import json
from data.base_loader import BaseLoader
from entities.raw_scenes import RawScenes

class TrajnetLoader(BaseLoader):
    def load_scenes(self) -> RawScenes:
        scenes = []
        tracks = []
        with open(self.path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'scene' in data:
                    scene_data = data['scene']
                    scenes.append(RawScenes.SceneData(
                        id=scene_data['id'],
                        person_id=scene_data['p'],
                        start_frame_number=scene_data['s'],
                        end_frame_number=scene_data['e'],
                        fps=scene_data['fps'],
                        tag=scene_data.get('tag')
                    ))
                elif 'track' in data:
                    track_data = data['track']
                    tracks.append(RawScenes.TrackData(
                        frame_number=track_data['f'],
                        person_id=track_data['p'],
                        x=track_data['x'],
                        y=track_data['y']
                    ))
        return RawScenes(scenes, tracks)