from data.loaders.base_loader import BaseLoader
from entities.raw_data import RawDataCollection, RawSceneData, RawSceneTrajectories, RawTrackData
from entities.vector2d import Point2D
from collections import defaultdict

class JuelichBottleneckLoader(BaseLoader):
    
    ROOM_SIZE = {"min": Point2D(x=-4, y=-4), "max": Point2D(x=4, y=5)}
    ROOM_BOUNDARIES = [Point2D(x=-4, y=-4), Point2D(x=-4, y=5), Point2D(x=4, y=5), Point2D(x=4, y=-4)]
    GOAL_POSITION = Point2D(
        x=ROOM_SIZE['max'].x + 1,  # 1 meter offset
        y=ROOM_SIZE['max'].y / 2
    )

    JUELICH_BOTTLENECK_FILE_INFO = {
        "b090": {'w': 0.9, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b100": {'w': 1.0, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b110": {'w': 1.1, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b120": {'w': 1.2, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b140": {'w': 1.4, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b160": {'w': 1.6, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b180": {'w': 1.8, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b200": {'w': 2.0, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b220": {'w': 2.2, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "b250": {'w': 2.5, 'l': 4.0, 'd': 4.0, 'w_f': 7.0},
        "l0": {'w': 1.2, 'l': 0.06, 'd': 4.0, 'w_f': 7.0},
        "l2": {'w': 1.2, 'l': 2.0, 'd': 4.0, 'w_f': 7.0},
        "l4": {'w': 1.2, 'l': 4.0, 'd': 4.0, 'w_f': 7.0}
    }

    def load_scenes_by_ids(self, scene_ids: set[int]) -> RawDataCollection:
        if len(scene_ids) > 1:
            raise ValueError(f'Juelich dataset only contains one scene')

        scene_id = next(iter(scene_ids), 1)

        return self._load_scene(self.path, self.dataset_name, scene_id)

    def load_all_scenes(self) -> RawDataCollection:
        return self._load_scene(self.path, self.dataset_name)

    def _load_scene(self, path: str, dataset_name: str, scene_id: int = 1) -> RawDataCollection:
        parsed_file = JuelichBottleneckLoader.parse_file(path)
        person_ids = list(set([x[0] for x in parsed_file]))
        frame_numbers = set([x[1] for x in parsed_file])
        start_frame_number = min(frame_numbers)
        end_frame_number = max(frame_numbers)

        trajectories: RawSceneTrajectories = defaultdict(dict)
        for person_id, frame_number, x, y in parsed_file:
            trajectories[person_id][frame_number] = RawTrackData(
                frame_number=frame_number,
                object_id=person_id,
                type="person",
                position=Point2D(x=x, y=y)
            )

        boundaries = self.get_boundaries(self.JUELICH_BOTTLENECK_FILE_INFO[dataset_name])

        raw_scene = RawSceneData(
            id=scene_id,
            goal_positions={pid: self.GOAL_POSITION for pid in person_ids},
            obstacles=boundaries,
            start_frame_number=start_frame_number,
            end_frame_number=end_frame_number,
            fps=25
        )

        return RawDataCollection(
            scenes=[raw_scene],
            dataset_name=dataset_name,
            trajectories={scene_id: trajectories}
        )

    def get_boundaries(self, wall_info: dict[str, float]) -> list[list[Point2D]]:
        room_width = self.ROOM_SIZE['max'].y - self.ROOM_SIZE['min'].y
        room_width_wo_bneck = room_width - wall_info['w']
        room_width_wo_bneck_oneside = room_width_wo_bneck / 2
        return [
            [
                self.ROOM_SIZE['min'],
                Point2D(x=self.ROOM_SIZE['max'].x - wall_info['l'], y=self.ROOM_SIZE['min'].y),
                Point2D(x=self.ROOM_SIZE['max'].x - wall_info['l'], y=self.ROOM_SIZE['min'].y + room_width_wo_bneck_oneside),
                Point2D(x=self.ROOM_SIZE['max'].x, y=self.ROOM_SIZE['min'].y + room_width_wo_bneck_oneside),
            ],
            [
                Point2D(x=self.ROOM_SIZE['min'].x, y=self.ROOM_SIZE['max'].y),
                Point2D(x=self.ROOM_SIZE['max'].x - wall_info['l'], y=self.ROOM_SIZE['max'].y),
                Point2D(x=self.ROOM_SIZE['max'].x - wall_info['l'], y=self.ROOM_SIZE['max'].y - room_width_wo_bneck_oneside),
                Point2D(x=self.ROOM_SIZE['max'].x, y=self.ROOM_SIZE['max'].y - room_width_wo_bneck_oneside),
            ],
        ]

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