import json
from data.base_loader import BaseLoader
from entities.raw_scenes_data import RawSceneData

class JuelichLoader(BaseLoader):
    def load_scene_by_id(self, scene_id: int) -> RawSceneData:
        pass