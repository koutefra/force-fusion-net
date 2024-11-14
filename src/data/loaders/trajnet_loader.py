import json
from data.loaders.base_loader import BaseLoader
from entities.raw_data import RawDataCollection, RawSceneData, RawPersonTrackData
from entities.vector2d import Point2D
from typing import Optional, Any

class TrajnetLoader(BaseLoader):
    def load_scenes_by_ids(self, scene_ids: set[int]) -> RawDataCollection:
        return self._load_scenes(self.path, self.dataset_name, scene_ids)

    def load_all_scenes(self) -> RawDataCollection:
        return self._load_scenes(self.path, self.dataset_name)

    @staticmethod
    def _load_scenes(
        path: str, 
        dataset_name: str,
        scene_ids: Optional[set[int]]
    ) -> tuple[list[RawSceneData], list[RawPersonTrackData]]:
        scenes = []
        tracks = []
        with open(path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'scene' in data: 
                    if not scene_ids or data['scene']['id'] in scene_ids:
                        scenes.append(TrajnetLoader.parse_scene(data, dataset_name))

                elif 'track' in data:
                    track_data = TrajnetLoader.parse_track(data)
                    if scene_ids:
                        for scene in scenes:
                            if scene.start_frame_number <= track_data['f'] <= scene.end_frame_number:
                                tracks.append(track_data)
                                break
                    else:
                        tracks.append(track_data)
        return RawDataCollection(
            scenes=scenes,
            dataset_name=dataset_name,
            tracks=tracks
        )
    
    @staticmethod
    def parse_scene(data: dict[Any], dataset_name: str) -> RawSceneData:
        """Parse scene data and return a RawSceneData instance."""
        scene = data['scene']
        return RawSceneData(
            id=scene['id'],
            goal_positions={},
            obstacles=[],
            start_frame_number=scene['s'],
            end_frame_number=scene['e'],
            fps=scene['fps'],
            tag=scene.get('tag'),
            dataset_name=dataset_name
        )

    @staticmethod
    def parse_track(data: dict[Any]) -> RawPersonTrackData:
        """Parse track data and return a RawPersonTrackData instance."""
        track = data['track']
        return RawPersonTrackData(
            frame_number=track['f'],
            person_id=track['p'],
            position=Point2D(x=track['x'], y=track['y'])
        )