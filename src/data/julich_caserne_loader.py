from data.base_loader import BaseLoader
from entities.vector2d import Point2D
from collections import defaultdict, OrderedDict
from entities.scene import Scene, Scenes
from entities.person import Person
from entities.obstacle import LineObstacle
from entities.frame import Frames, Trajectories, Trajectory
from tqdm import tqdm

class JulichCaserneLoader(BaseLoader):
    # fps should be 12.5, according to this: https://arxiv.org/pdf/physics/0702004
    fps = 12.5
    frame_step = 1
    goal_position = Point2D(x=4.9, y=0.0)
    bounding_box = (Point2D(x=-4.0, y=-4.0), Point2D(x=5.0, y=4.0))

    BOUNDARIES = {
        "synth1": "POLYGON ((-3.5 -3.5, -3.5 1, 2.5 1, 2.5 4, 2.45 4, 2.45 1.05, -3.55 1.05, -3.55 -3.5), (-2.5 -3.5, -2.5 0, 3.5 0, 3.5 4, 3.55 4, 3.55 -0.05, -2.45 -0.05, -2.45 -3.5)",
        "b090": "POLYGON ((-3.5 -3.5, -3.5 0, -0.45 0, -0.45 4, -0.5 4, -0.5 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.45 0, 0.45 4, 0.5 4, 0.5 0.05, 3.55 0.05, 3.55 -3.5))",
        "b100": "POLYGON ((-3.5 -3.5, -3.5 0, -0.5 0, -0.5 4, -0.55 4, -0.55 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.5 0, 0.5 4, 0.55 4, 0.55 0.05, 3.55 0.05, 3.55 -3.5))",
        "b110": "POLYGON ((-3.5 -3.5, -3.5 0, -0.55 0, -0.55 4, -0.6 4, -0.6 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.55 0, 0.55 4, 0.6 4, 0.6 0.05, 3.55 0.05, 3.55 -3.5))",
        "b120": "POLYGON ((-3.5 -3.5, -3.5 0, -0.6 0, -0.6 4, -0.65 4, -0.65 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 4, 0.65 4, 0.65 0.05, 3.55 0.05, 3.55 -3.5))",
        "b140": "POLYGON ((-3.5 -3.5, -3.5 0, -0.7 0, -0.7 4, -0.75 4, -0.75 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.7 0, 0.7 4, 0.75 4, 0.75 0.05, 3.55 0.05, 3.55 -3.5))",
        "b160": "POLYGON ((-3.5 -3.5, -3.5 0, -0.8 0, -0.8 4, -0.85 4, -0.85 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.8 0, 0.8 4, 0.85 4, 0.85 0.05, 3.55 0.05, 3.55 -3.5))",
        "b180": "POLYGON ((-3.5 -3.5, -3.5 0, -0.9 0, -0.9 4, -0.95 4, -0.95 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.9 0, 0.9 4, 0.95 4, 0.95 0.05, 3.55 0.05, 3.55 -3.5))",
        "b200": "POLYGON ((-3.5 -3.5, -3.5 0, -1 0, -1 4, -1.05 4, -1.05 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 1 0, 1 4, 1.05 4, 1.05 0.05, 3.55 0.05, 3.55 -3.5))",
        "b220": "POLYGON ((-3.5 -3.5, -3.5 0, -1.1 0, -1.1 4, -1.15 4, -1.15 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 1.1 0, 1.1 4, 1.15 4, 1.15 0.05, 3.55 0.05, 3.55 -3.5))",
        "b250": "POLYGON ((-3.5 -3.5, -3.5 0, -1.25 0, -1.25 4, -1.3 4, -1.3 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 1.25 0, 1.25 4, 1.3 4, 1.3 0.05, 3.55 0.05, 3.55 -3.5))",
        "l0": "POLYGON ((-3.5 -3.5, -3.5 0, -0.6 0, -0.6 0.06, -1.3 0.06, -1.3 4, -1.35 4, -1.35 4, -1.35 0.06, -3.55 0.06, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 0.06, 1.3 0.06, 1.3 4, 1.35 4, 1.35 4, 1.35 0.06, 3.55 0.06, 3.55 -3.5))",
        "l2": "POLYGON ((-3.5 -3.5, -3.5 0, -0.6 0, -0.6 2.1, -1.3 4, -1.35 4, -0.65 2.1, -0.65 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 2.1, 1.3 4, 1.35 4, 0.65 2.1, 0.65 0.05, 3.55 0.05, 3.55 -3.5))",
        "l4": "POLYGON ((-3.5 -3.5, -3.5 0, -0.6 0, -0.6 4, -0.65 4, -0.65 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 4, 0.65 4, 0.65 0.05, 3.55 0.05, 3.55 -3.5))"
    }

    def __init__(self, paths_and_names: list[tuple[str, str]]):
        self.paths_and_names = paths_and_names
    
    def load(self, print_progress: bool = True) -> Scenes:
        scenes = {}
        for path, name in tqdm(
            self.paths_and_names, 
            desc=f"Loading scenes for juelich_bneck dataset...", 
            disable=not print_progress):
            scene = self._load_scene(path, name)
            scenes[scene.id] = scene
        return Scenes(scenes)

    def _load_scene(
        self,
        path: str, 
        scene_name: str, 
    ) -> Scene:
        parsed_file = JulichCaserneLoader.parse_file(path)

        trajectories_raw = defaultdict(lambda: OrderedDict())
        for person_id, frame_number, x, y in parsed_file:
            trajectories_raw[person_id][frame_number] = Person(
                id=person_id,
                position=Point2D(x=x*0.01, y=y*0.01),  # conversion from cm to meters
                goal=self.goal_position
            )
        trajectories = Trajectories({k: Trajectory(person_id=k, records=v) for k, v in trajectories_raw.items()})
        frame_numbers = list(sorted(set([f_num for t in trajectories.values() for f_num in t.records.keys()])))

        obstacles = [
            LineObstacle(p1=obstacle[i], p2=obstacle[i + 1]) 
            for obstacle in JulichCaserneLoader.parse_polygon_string(JulichCaserneLoader.BOUNDARIES[scene_name])
            for i in range(len(obstacle) - 1)
        ]
        obstacles_dict = {f_num: obstacles for f_num in frame_numbers}

        return Scene(
            id=f"juelich_{scene_name}",
            frames=Frames.from_trajectories(trajectories, obstacles_dict),
            bounding_box=self.bounding_box,
            frame_step=self.frame_step,
            fps=self.fps
        )

    @staticmethod
    def parse_file(path: str) -> list[tuple[int, int, float, float]]:
        parsed_data = []
        with open(path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                person_id = int(parts[0])
                frame_number = int(parts[1])
                position_y = float(parts[2])
                position_x = float(parts[3])
                parsed_data.append((person_id, frame_number, position_x, position_y))
        return parsed_data

    @staticmethod
    def parse_polygon_string(polygon_str: str) -> list[list[Point2D]]:
        def parse_coordinates(coord_str: str) -> list[Point2D]:
            coord_pairs = coord_str.strip("()").split(", ")
            points = [Point2D(float(y), float(x)) for x, y in (pair.split() for pair in coord_pairs)]
            return points
        ring_strs = polygon_str.replace("POLYGON ((", "").replace("))", "").split("), (")
        rings = [parse_coordinates(ring) for ring in ring_strs]
        return rings