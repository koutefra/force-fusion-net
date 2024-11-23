from entities.vector2d import Point2D, Velocity, Acceleration 
from entities.features import Features, IndividualFeatures, InteractionFeatures, ObstacleFeatures
import math

class SocialForceModel:
    def __init__(
        self, 
        param_valus_to_cm: bool, 
        A_interaction: float = 2.0,  # Interaction force constant, (m/s^(-2)) 
        A_obstacle: float = 6.0,  # Interaction force constant, (m/s^(-2)) 
        B_interaction: float = 0.3,  # Interaction decay constant, m
        B_obstacle: float = 0.5,  # Interaction decay constant, m
        tau: float = 0.3,  # Relaxation time constant, s
        desired_speed: float = 0.8,  # m/s
    ):
        distance_scale = 100 if param_valus_to_cm else 1
        self.A_interaction = A_interaction * distance_scale
        self.A_obstacle = A_obstacle * distance_scale
        self.B_interaction = B_interaction * distance_scale
        self.B_obstacle = B_obstacle * distance_scale
        self.tau = tau
        self.desired_speed = desired_speed * distance_scale

    def _desired_force(self, f: IndividualFeatures) -> Acceleration:
        dir_to_goal = Point2D(x=f.dir_x_to_goal, y=f.dir_y_to_goal)
        velocity = Velocity(x=f.vel_x, y=f.vel_y)
        desired_velocity = dir_to_goal * self.desired_speed
        desired_acceleration = (desired_velocity - velocity) * (1 / self.tau)
        return Acceleration(desired_acceleration.x, desired_acceleration.y)

    def _compute_force(self, features: list[tuple[Point2D, float]], A: float, B: float) -> Acceleration:
        """Compute the total force exerted by multiple other positions on a person."""
        total_force_x = 0.0
        total_force_y = 0.0

        for direction, distance in features:
            force_magnitude = A * math.exp(-distance / B)
            total_force_x += - direction.x * force_magnitude
            total_force_y += - direction.y * force_magnitude

        return Acceleration(x=total_force_x, y=total_force_y)

    def _interaction_force(self, fs: list[InteractionFeatures]) -> Acceleration:
        """Compute the total interaction force from all other pedestrians."""
        features = [
            (Point2D(x=f.dir_x_to_p, y=f.dir_y_to_p), f.dist_to_p)
            for f in fs
        ]
        return self._compute_force(features, self.A_interaction, self.B_interaction)

    def _obstacle_force(self, fs: list[ObstacleFeatures]) -> Acceleration:
        """Compute the total force exerted by all obstacles."""
        features = [
            (Point2D(x=f.dir_x_to_o_cls, y=f.dir_y_to_o_cls), f.dist_to_o_cls)
            for f in fs
        ]
        return self._compute_force(features, self.A_obstacle, self.B_obstacle)

    def predict(self, features: list[Features]) -> list[Acceleration]: 
        preds_acc = []
        for f in features:
            desired_force = self._desired_force(f.individual_features)
            interaction_force = self._interaction_force(f.interaction_features)
            obstacle_force = self._obstacle_force(f.obstacle_features)
            total_force = desired_force + interaction_force + obstacle_force
            preds_acc.append(total_force)
        return preds_acc