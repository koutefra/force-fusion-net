from dataclasses import dataclass
from entities.vector2d import Point2D, Velocity, Acceleration
from entities.vector2d import kinematic_equation
from typing import Optional, Callable

@dataclass(frozen=True)
class Person:
    id: int
    position: Point2D
    goal: Point2D
    velocity: Optional[Velocity] = None
    acceleration: Optional[Acceleration] = None

    def normalized(
        self, 
        pos_scale: Callable[[Point2D], Point2D] = lambda p: p, 
        vel_scale: Callable[[Velocity], Velocity] = lambda v: v, 
        acc_scale: Callable[[Acceleration], Acceleration] = lambda a: a) -> "Person":
        return Person(
            id=self.id,
            position=pos_scale(self.position),
            goal=pos_scale(self.goal),
            velocity=vel_scale(self.velocity) if self.velocity else None,
            acceleration=acc_scale(self.acceleration) if self.acceleration else None
        )
    
    def apply_kinematic_equation(self, delta_time: float) -> "Person":
        next_position, next_velocity= kinematic_equation(
            cur_positions=self.position,
            cur_velocities=self.velocity,
            delta_times=delta_time,
            cur_accelerations=self.acceleration
        )
        return Person(
            id=self.id,
            position=next_position,
            goal=self.goal,
            velocity=next_velocity
        )

    def set_velocity(self, vel: Velocity) -> "Person":
        return Person(
            id=self.id,
            position=self.position,
            goal=self.goal,
            velocity=vel,
            acceleration=self.acceleration
        )

    def set_acceleration(self, acc: Acceleration) -> "Person":
        return Person(
            id=self.id,
            position=self.position,
            goal=self.goal,
            velocity=self.velocity,
            acceleration=acc
        )

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