from entities.vector2d import Point2D, Velocity, Acceleration 
from entities.scene import Scene
from entities.frame_object import FrameObject, PersonInFrame
from sklearn.model_selection import ParameterGrid
import math

class SocialForceModel:
    def __init__(
        self, 
        A: float = 1.5, 
        B: float = 0.4, 
        tau: float = 1.0, 
        radius: float = 1.5, 
        desired_speed: float = 1.5
    ):
        self.A = A  # Interaction force constant
        self.B = B  # Interaction decay constant
        self.radius = radius  # Interaction radius
        self.tau = tau  # Relaxation time constant
        self.desired_speed = desired_speed

    def _desired_force(self, cur_pos: Point2D, goal_pos: Point2D, velocity: Velocity) -> Acceleration:
        direction = goal_pos - cur_pos
        desired_velocity = direction.normalize() * self.desired_speed
        desired_acceleration = (desired_velocity - velocity) * (1 / self.tau)
        return Acceleration(desired_acceleration.x, desired_acceleration.y)

    def _interaction_force(self, pos_1: Point2D, pos_2: Point2D) -> Acceleration:
        direction = pos_1 - pos_2
        distance = direction.magnitude()
        force_magnitude = self.A * math.exp((self.radius - distance) / self.B)
        interaction_acceleration = direction.normalize() * force_magnitude
        return Acceleration(interaction_acceleration.x, interaction_acceleration.y)

    def _compute_interaction_forces(self, person: PersonInFrame, frame_objs: list[FrameObject]) -> Acceleration:
        """Compute the sum of interaction forces from all other pedestrians."""
        total_interaction_force_x = 0.0
        total_interaction_force_y = 0.0

        for frame_obj in frame_objs:
            if not isinstance(frame_obj, PersonInFrame):
                continue

            other_person = frame_obj
            if person.id == other_person.id:
                continue

            interaction_force = self._interaction_force(person.position, other_person.position)
            total_interaction_force_x += interaction_force.x
            total_interaction_force_y += interaction_force.y
        return Acceleration(x=total_interaction_force_x, y=total_interaction_force_y)

    def predict_frame(self, person: PersonInFrame, frame_objs: list[FrameObject], goal_pos: Point2D) -> Acceleration: 
        desired_force = self._desired_force(person.position, goal_pos, person.velocity)
        interaction_force = self._compute_interaction_forces(person, frame_objs)
        total_force = desired_force + interaction_force
        return total_force

    def predict_scene(self, scene: Scene) -> list[Acceleration]:
        predicted_forces = []
        for frame in scene.frames:
            focus_person = next(
                (obj for obj in frame.frame_objects if isinstance(obj, PersonInFrame) and obj.id == scene.focus_person_id),
                None
            )
            if focus_person is None:
                continue  # Skip if focus person not found in frame

            predicted_force = self.predict_frame(focus_person, frame.frame_objects, scene.focus_person_goal)
            predicted_forces.append(predicted_force)
        return predicted_forces

    def predict_scenes(self, scenes: dict[int, Scene]) -> dict[int, list[Acceleration]]:
        predicted_forces = {}
        for scene_id, scene in scenes.items():
            predicted_forces[scene_id] = self.predict_scene(scene)
        return predicted_forces