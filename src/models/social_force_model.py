from entities.vector2d import Point2D, Velocity, Acceleration 
from entities.features import Features, IndividualFeatures, InteractionFeatures, ObstacleFeatures
import math

class SocialForceModel:
    def __init__(
        self, 
        param_valus_to_cm: bool, 
        A: float = 3.0,  # Interaction force constant, (m/s^(-2)) 
        B: float = 0.3,  # Interaction decay constant, m
        tau: float = 0.3,  # Relaxation time constant, s
        desired_speed: float = 0.8,  # m/s
    ):
        distance_scale = 100 if param_valus_to_cm else 1

        self.A = A * distance_scale
        self.B = B * distance_scale
        self.tau = tau
        self.desired_speed = desired_speed * distance_scale

    def _desired_force(self, f: IndividualFeatures) -> Acceleration:
        dir_to_goal = Point2D(x=f.direction_x_to_goal, y=f.direction_y_to_goal)
        velocity = Velocity(x=f.velocity_x, y=f.velocity_y)
        desired_velocity = dir_to_goal * self.desired_speed
        desired_acceleration = (desired_velocity - velocity) * (1 / self.tau)
        return Acceleration(desired_acceleration.x, desired_acceleration.y)

    def _compute_force(self, features: list[tuple[Point2D, float]]) -> Acceleration:
        """Compute the total force exerted by multiple other positions on a person."""
        total_force_x = 0.0
        total_force_y = 0.0

        for direction, distance in features:
            force_magnitude = self.A * math.exp(-distance / self.B)
            total_force_x += - direction.x * force_magnitude
            total_force_y += - direction.y * force_magnitude

        return Acceleration(x=total_force_x, y=total_force_y)

    def _interaction_force(self, fs: list[InteractionFeatures]) -> Acceleration:
        """Compute the total interaction force from all other pedestrians."""
        features = [
            (Point2D(x=f.direction_x_to_person_p, y=f.direction_y_to_person_p), f.distance_to_person_p)
            for f in fs
        ]
        return self._compute_force(features)

    def _obstacle_force(self, fs: list[ObstacleFeatures]) -> Acceleration:
        """Compute the total force exerted by all obstacles."""
        features = [
            (Point2D(x=f.direction_x_to_obstacle_o, y=f.direction_y_to_obstacle_o), f.distance_to_obstacle_o)
            for f in fs
        ]
        return self._compute_force(features)

    def predict(self, features: list[Features]) -> list[Acceleration]: 
        preds_acc = []
        for f in features:
            desired_force = self._desired_force(f.individual_features)
            interaction_force = self._interaction_force(f.interaction_features)
            obstacle_force = self._obstacle_force(f.obstacle_features)
            total_force = desired_force + interaction_force + obstacle_force
            print(desired_force.magnitude(), interaction_force.magnitude(), obstacle_force.magnitude())
            preds_acc.append(total_force)
        return preds_acc