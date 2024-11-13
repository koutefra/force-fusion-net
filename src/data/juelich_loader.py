from data.base_loader import BaseLoader
from entities.raw_data import RawDataCollection, RawSceneData, RawSceneTrajectories
from entities.vector2d import Point2D

class JuelichLoader(BaseLoader):
    def load_scene_by_ids(self, scene_ids: set[int]) -> RawDataCollection:
        return self.load_scenes_by_ids(scene_ids=scene_ids)

    def load_all_scenes(self) -> RawDataCollection:
        pass

    def load_scene(self, path: str, scene_id: int) -> RawDataCollection:
        parsed_file = self.parse_file(path)
        person_ids = list(set([x[0] for x in parsed_file]))
        frame_numbers = set([x[1] for x in parsed_file])
        start_frame_number = min(frame_numbers)
        end_frame_number = max(frame_numbers)
        x = RawSceneData(
            id=scene_id,
            focus_person_ids=person_ids,
            goal_positions=Point2D.zero(),
            start_frame_number=start_frame_number,
            end_frame_number=end_frame_number,
            fps=
        )
        return None




    def parse_file(self, path: str) -> list[tuple[int, int, float, float]]:
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