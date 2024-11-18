from data.loaders.base_loader import BaseLoader
from entities.vector2d import Point2D
from collections import defaultdict, OrderedDict
from entities.scene import Scene, Scenes, Trajectories, Person
from entities.obstacle import LineObstacle
from data.fdm_calculator import FiniteDifferenceCalculator
from typing import Optional

class JuelichBneckLoader(BaseLoader):
    def __init__(
        self, 
        paths_and_names: list[tuple[str, str]], 
        fdm_calculator: Optional[FiniteDifferenceCalculator]
    ):
        self.paths_and_names = paths_and_names
        self.fdm_calculator = fdm_calculator
    
    BOUNDARIES = {
        "b090": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.45 0, -0.45 4, -0.5 4, -0.5 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.45 0, 0.45 4, 0.5 4, 0.5 0.05, 3.55 0.05, 3.55 -3.5))",
        "b100": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.5 0, -0.5 4, -0.55 4, -0.55 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.5 0, 0.5 4, 0.55 4, 0.55 0.05, 3.55 0.05, 3.55 -3.5))",
        "b110": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.55 0, -0.55 4, -0.6 4, -0.6 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.55 0, 0.55 4, 0.6 4, 0.6 0.05, 3.55 0.05, 3.55 -3.5))",
        "b120": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.6 0, -0.6 4, -0.65 4, -0.65 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 4, 0.65 4, 0.65 0.05, 3.55 0.05, 3.55 -3.5))",
        "b140": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.7 0, -0.7 4, -0.75 4, -0.75 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.7 0, 0.7 4, 0.75 4, 0.75 0.05, 3.55 0.05, 3.55 -3.5))",
        "b160": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.8 0, -0.8 4, -0.85 4, -0.85 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.8 0, 0.8 4, 0.85 4, 0.85 0.05, 3.55 0.05, 3.55 -3.5))",
        "b180": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.9 0, -0.9 4, -0.95 4, -0.95 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.9 0, 0.9 4, 0.95 4, 0.95 0.05, 3.55 0.05, 3.55 -3.5))",
        "b200": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -1 0, -1 4, -1.05 4, -1.05 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 1 0, 1 4, 1.05 4, 1.05 0.05, 3.55 0.05, 3.55 -3.5))",
        "b220": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -1.1 0, -1.1 4, -1.15 4, -1.15 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 1.1 0, 1.1 4, 1.15 4, 1.15 0.05, 3.55 0.05, 3.55 -3.5))",
        "b250": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -1.25 0, -1.25 4, -1.3 4, -1.3 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 1.25 0, 1.25 4, 1.3 4, 1.3 0.05, 3.55 0.05, 3.55 -3.5))",
        "l0": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.6 0, -0.6 0.06, -1.3 0.06, -1.3 4, -1.35 4, -1.35 4, -1.35 0.06, -3.55 0.06, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 0.06, 1.3 0.06, 1.3 4, 1.35 4, 1.35 4, 1.35 0.06, 3.55 0.06, 3.55 -3.5))",
        "l2": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.6 0, -0.6 2.1, -1.3 4, -1.35 4, -0.65 2.1, -0.65 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 2.1, 1.3 4, 1.35 4, 0.65 2.1, 0.65 0.05, 3.55 0.05, 3.55 -3.5))",
        "l4": "POLYGON ((-4 -4, -4 5, 4 5, 4 -4), (-3.5 -3.5, -3.5 0, -0.6 0, -0.6 4, -0.65 4, -0.65 0.05, -3.55 0.05, -3.55 -3.5), (3.5 -3.5, 3.5 0, 0.6 0, 0.6 4, 0.65 4, 0.65 0.05, 3.55 0.05, 3.55 -3.5))"
    }

    def load(self) -> Scenes:
        scenes = {}
        for path, name in self.paths_and_names:
            scene = self._load_scene(path, name)
            scenes[name] = scene
        return scenes

    @staticmethod
    def _load_scene(
        path: str, 
        dataset_name: str, 
        goal_position: Point2D = Point2D(x=499, y=200),
        fdm_calculator: Optional[FiniteDifferenceCalculator] = None
    ) -> Scene:
        parsed_file = JuelichBneckLoader.parse_file(path)

        trajectories: Trajectories = defaultdict(lambda: OrderedDict())
        for person_id, frame_number, x, y in parsed_file:
            trajectories[person_id][frame_number] = Person(
                position=Point2D(x=x, y=y),
                goal=goal_position
            )

        if fdm_calculator:
            fdm_calculator.compute_velocities(trajectories)
            fdm_calculator.compute_accelerations(trajectories)
            
        obstacles = [
            # time 100 for conversion from meters to centimeters
            LineObstacle(obstacle[i] * 100, obstacle[i + 1] * 100) 
            for obstacle in JuelichBneckLoader.parse_polygon_string(JuelichBneckLoader.BOUNDARIES[dataset_name])
            for i in range(len(obstacle) - 1)
        ]

        return Scene(
            id=dataset_name,
            obstacles=obstacles,
            frames=Scene.trajectories_to_frames(trajectories),
            fps=25.0
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
