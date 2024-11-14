from data.loaders.base_loader import BaseLoader
from entities.raw_data import RawDataCollection, RawSceneData, RawSceneTrajectories, RawTrackData
from entities.vector2d import Point2D
from collections import defaultdict

class JuelichLoader(BaseLoader):
    def load_scenes_by_ids(self, scene_ids: set[int]) -> RawDataCollection:
        if len(scene_ids) > 1:
            raise ValueError(f'Juelich dataset only contains one scene')

        scene_id = next(iter(scene_ids), 1)

        return self._load_scene(self.path, self.dataset_name, scene_id)

    def load_all_scenes(self) -> RawDataCollection:
        return self._load_scene(self.path, self.dataset_name)

    @staticmethod
    def _load_scene(path: str, dataset_name: str, scene_id: int = 1) -> RawDataCollection:
        parsed_file = JuelichLoader.parse_file(path)
        person_ids = list(set([x[0] for x in parsed_file]))
        frame_numbers = set([x[1] for x in parsed_file])
        start_frame_number = min(frame_numbers)
        end_frame_number = max(frame_numbers)

        trajectories: RawSceneTrajectories = defaultdict(dict)
        for person_id, frame_number, x, y in parsed_file:
            trajectories[person_id][frame_number] = RawTrackData(
                frame_number=frame_number,
                object_id=person_id,
                position=Point2D(x=x, y=y)
            )

        raw_scene = RawSceneData(
            id=scene_id,
            focus_person_ids=person_ids,
            goal_positions={person_id: Point2D.zero() for person_id in person_ids},
            start_frame_number=start_frame_number,
            end_frame_number=end_frame_number,
            fps=25
        )

        return RawDataCollection(
            scenes=[raw_scene],
            dataset_name=dataset_name,
            trajectories={scene_id: trajectories}
        )

    @staticmethod
    def parse_file(path: str) -> list[tuple[int, int, float, float]]:
        parsed_data = []
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                person_id = int(parts[0])
                frame_number = int(parts[1])
                position_x = float(parts[2])
                position_y = float(parts[3])
                parsed_data.append((person_id, frame_number, position_x, position_y))
        return parsed_data