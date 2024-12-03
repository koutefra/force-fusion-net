from dataclasses import dataclass
from entities.vector2d import Point2D, Velocity, Acceleration, closest_point_on_line
from entities.obstacle import LineObstacle
from typing import Optional

@dataclass(frozen=True)
class Person:
    id: int
    position: Point2D
    goal: Point2D
    velocity: Optional[Velocity] = None
    acceleration: Optional[Acceleration] = None

    def get_individual_features(self) -> list[float]:
        dist_to_goal = (self.position - self.goal).magnitude()
        dir_to_goal = self.position.direction_to(self.goal)
        vel_towards_goal = self.velocity.dot(dir_to_goal)
        return [
            self.velocity.x,
            self.velocity.y,
            dist_to_goal,
            dir_to_goal.x,
            dir_to_goal.y,
            vel_towards_goal
        ]

    def get_obstacle_features(self, obstacles: list[LineObstacle]) -> list[list[float]]:
        obstacle_features = []
        for line in obstacles:
            closest_point = closest_point_on_line(self.position, line.p1, line.p2)

            distances, directions = {}, {}
            for name, point in [('closest', closest_point), ('start', line.p1), ('end', line.p2)]:
                distances[name] = (self.position - point).magnitude()
                directions[name] = self.position.direction_to(point)

            obstacle_features.append([
                distances['closest'],
                directions['closest'].x,
                directions['closest'].y,
                distances['start'],
                directions['start'].x,
                directions['start'].y,
                distances['end'],
                directions['end'].x,
                directions['end'].y
            ])
        return obstacle_features