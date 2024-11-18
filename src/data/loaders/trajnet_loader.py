import json
from data.loaders.base_loader import BaseLoader
from entities.vector2d import Point2D
from typing import Optional, Any
from data.fdm_calculator import FiniteDifferenceCalculator
from entities.scene import Scene, Scenes

class TrajnetLoader(BaseLoader):
    def __init__(self, path: str, dataset_name: str, fdm_calculator: FiniteDifferenceCalculator):
        super().__init__(path, dataset_name)
        self.fdm_calculator = fdm_calculator
        
    INFO = {
        "biwi_hotel": {"n_scenes": 229, "n_tracks": 4621, "frame_freq": 10},
        "cff_06": {"n_scenes": 23751, "n_tracks": 2229471, "frame_freq": 4},
        "cff_07": {"n_scenes": 24100, "n_tracks": 2275581, "frame_freq": 4},
        "cff_08": {"n_scenes": 23070, "n_tracks": 2114979, "frame_freq": 4},
        "cff_09": {"n_scenes": 11433, "n_tracks": 974323, "frame_freq": 4},
        "cff_10": {"n_scenes": 9702, "n_tracks": 826929, "frame_freq": 4},
        "cff_12": {"n_scenes": 23936, "n_tracks": 2221716, "frame_freq": 4},
        "cff_13": {"n_scenes": 22355, "n_tracks": 2041636, "frame_freq": 4},
        "cff_14": {"n_scenes": 23376, "n_tracks": 2122967, "frame_freq": 4},
        "cff_15": {"n_scenes": 22657, "n_tracks": 2071108, "frame_freq": 4},
        "cff_16": {"n_scenes": 10771, "n_tracks": 907048, "frame_freq": 4},
        "cff_17": {"n_scenes": 10000, "n_tracks": 856922, "frame_freq": 4},
        "cff_18": {"n_scenes": 22021, "n_tracks": 2011313, "frame_freq": 4},
        "crowds_students001": {"n_scenes": 5265, "n_tracks": 26600, "frame_freq": 10},
        "crowds_students003": {"n_scenes": 4262, "n_tracks": 21740, "frame_freq": 10},
        "crowds_zara01": {"n_scenes": 1017, "n_tracks": 4965, "frame_freq": 10},
        "crowds_zara03": {"n_scenes": 955, "n_tracks": 4825, "frame_freq": 10},
        "lcas": {"n_scenes": 889, "n_tracks": 10499, "frame_freq": 1},
        "wildtrack": {"n_scenes": 1101, "n_tracks": 9518, "frame_freq": 5}
    }

    def load(self) -> Scenes:
        pass

    def _load_scenes(
        self,
        scene_ids: Optional[set[int]] = None
    ) -> ...:
        scenes = {}
        tracks = []
        with open(self.path, 'r') as file:
            for line in file:
                data = json.loads(line)
                if 'scene' in data: 
                    if not scene_ids or data['scene']['id'] in scene_ids:
                        scenes[data['scene']['id']] = TrajnetLoader.parse_scene(data)

                elif 'track' in data:
                    track_data = TrajnetLoader.parse_track(data)
                    if scene_ids:
                        for scene in scenes.values():
                            if scene.start_frame_number <= track_data['f'] <= scene.end_frame_number:
                                tracks.append(track_data)
                                break
                    else:
                        tracks.append(track_data)
        return None
    
    @staticmethod
    def parse_scene(data: dict[Any]) -> ...:
        """Parse scene data and return a RawSceneData instance."""
        scene = data['scene']
        return None

    @staticmethod
    def parse_track(data: dict[Any]) -> ...:
        """Parse track data and return a RawPersonTrackData instance."""
        track = data['track']
        return None

    # @staticmethod
    # def _build_scene_trajectories(
    #     scenes: list[RawSceneData],
    #     tracks: list[RawPersonTrackData]
    # ) -> dict[int, RawSceneTrajectories]:
    #     trajectories = defaultdict(lambda: defaultdict(dict))
        
    #     # Create a lookup table to map frame numbers to relevant scenes
    #     frame_to_scene_ids = defaultdict(list)
    #     for scene in scenes.values():
    #         for frame in range(scene.start_frame_number, scene.end_frame_number + 1):
    #             frame_to_scene_ids[frame].append(scene.id)

    #     # Assign tracks to scenes based on frame numbers
    #     for track in tracks:
    #         relevant_scene_ids = frame_to_scene_ids.get(track.frame_number, [])
    #         for scene_id in relevant_scene_ids:
    #             trajectories[scene_id][track.person_id].setdefault(track.frame_number, track)

    #     # Sort trajectories by frame_number within each person's data
    #     for scene_trajectories in trajectories.values():
    #         for person_id, frames in scene_trajectories.items():
    #             sorted_frames = OrderedDict(sorted(frames.items()))
    #             scene_trajectories[person_id] = sorted_frames

    #     return trajectories