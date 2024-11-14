import pygame
from pygame import Surface
from entities.vector2d import Point2D
import math
import sys
from typing import Tuple
from entities.scene import Scene
from entities.frame_object import PersonInFrame

class Visualizer:
    default_colors: dict[str, tuple[int, int, int]] = {
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "medium_blue": (100, 149, 237),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "gray": (128, 128, 128),
        "dark_gray": (64, 64, 64),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "brown": (165, 42, 42),
        "pink": (255, 192, 203),
        "grass_light": (124, 252, 0),
        "grass_dark": (34, 139, 34),
        "skin_orange": (255, 165, 100),
        "red_orange": (255, 100, 0),
        "sky_blue": (135, 206, 235),
    }

    background_color = default_colors["grass_dark"]
    focus_person_color = default_colors["red_orange"]
    other_person_color = default_colors["sky_blue"]
    velocity_color = default_colors["blue"]
    acceleration_color = default_colors["red"]

    def __init__(
        self, target_screen_size: int = 1000, 
        circle_radius: int = 10, 
        text_size: int = 15,
        line_width: int = 3
    ):
        self.target_screen_size = target_screen_size
        self.circle_radius = circle_radius
        self.text_size = text_size
        self.line_width = line_width

    @staticmethod
    def draw_arrowhead(screen: Surface, start_pos: Point2D, end_pos: Point2D, color: Tuple[int, int, int]):
        """Draw an arrowhead at the end of a line."""
        angle = math.atan2(end_pos.y - start_pos.y, end_pos.x - start_pos.x)
        arrow_size = 10
        arrow_angle = math.pi / 6

        left_arrowhead = (
            end_pos.x - arrow_size * math.cos(angle + arrow_angle),
            end_pos.y - arrow_size * math.sin(angle + arrow_angle)
        )
        right_arrowhead = (
            end_pos.x - arrow_size * math.cos(angle - arrow_angle),
            end_pos.y - arrow_size * math.sin(angle - arrow_angle)
        )

        pygame.draw.line(screen, color, (end_pos.x, end_pos.y), left_arrowhead, 2)
        pygame.draw.line(screen, color, (end_pos.x, end_pos.y), right_arrowhead, 2)

    @staticmethod
    def draw_labels(screen: Surface, labels: list[tuple[str, tuple[int, int, int]]], font: pygame.font.SysFont):
        """Draw color labels for the visualization."""
        for i, (label, color) in enumerate(labels):
            label_surface = font.render(label, True, color)
            screen.blit(label_surface, (10, 10 + i * 20))

    def draw_scene(
        self, 
        scene: Scene, 
        time_scale: float = 1.0, 
        preds: list[tuple[str, list[PersonInFrame], tuple[int, int, int] | None]] = []
    ):
        min_pos, max_pos = scene.bounding_box
        bbox_size = max_pos - min_pos
        max_size = max(bbox_size)
        spatial_scale = self.target_screen_size / max_size

        # initialize pygame
        pygame.init()
        screen_res = (bbox_size * spatial_scale).to_int_tuple()
        screen = pygame.display.set_mode(screen_res)
        screen.fill(self.background_color)
        pygame.display.flip()
        clock = pygame.time.Clock()
        scaled_fps = int(scene.fps * time_scale)
        self.font = pygame.font.SysFont(None, self.text_size)
        pygame.display.set_caption(
            f"Scene {scene.id}; Scale 1:{spatial_scale:.0f}; Res: {screen_res[0]}x{screen_res[1]}" 
        )
        labels = [
            ("Focus Person", self.focus_person_color),
            ("Other Person", self.other_person_color),
            ("Velocity", self.velocity_color),
            ("Acceleration", self.acceleration_color),
        ] + [(x[0], x[2]) for x in preds]

        for frame_id, frame in enumerate(scene.frames):
            frame_pred_objects_and_colors = [(x[1][frame_id], x[2]) for x in preds]  # list[tuple[PersonInFrame, color]]
            frame_objects_and_colors = [
                (o, self.other_person_color) if not isinstance(o, PersonInFrame) # or o.id != scene.focus_person_id 
                else (o, self.focus_person_color)
                for o in frame.frame_objects
            ]
            for frame_obj, color in frame_pred_objects_and_colors + frame_objects_and_colors:
                if not isinstance(frame_obj, PersonInFrame):
                    continue

                person = frame_obj

                scaled_position = (person.position - min_pos) * spatial_scale
                pygame.draw.circle(screen, color, scaled_position.to_tuple(), self.circle_radius)

                # if person.id == scene.focus_person_id:
                #     # draw the goal position
                #     scaled_goal_position = (scene.focus_person_goal - min_pos) * spatial_scale
                #     pygame.draw.rect(
                #         screen, color, 
                #         pygame.Rect(
                #             (scaled_goal_position - self.circle_radius).to_tuple(),
                #             (self.circle_radius * 2, self.circle_radius * 2)
                #         )
                #     )

                scaled_velocity = 0.5 * (person.velocity * spatial_scale * time_scale)
                scaled_acceleration = 0.5 * (person.acceleration * spatial_scale * (time_scale ** 2))

                # draw velocity and acceleration vectors
                vectors = [
                    (scaled_velocity, self.velocity_color, "velocity"),
                    (scaled_acceleration, self.acceleration_color, "acceleration"),
                ]
                for vector, color, label in vectors:
                    pygame.draw.line(
                        screen, 
                        color, 
                        scaled_position.to_tuple(), 
                        (scaled_position + vector).to_tuple(),
                        self.line_width
                    )
                    self.draw_arrowhead(
                        screen, 
                        scaled_position, 
                        scaled_position + vector, 
                        color
                    )

            Visualizer.draw_labels(screen, labels, self.font)
            pygame.display.flip()
            clock.tick(scaled_fps)
            screen.fill(self.background_color)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def visualize(
        self, 
        scene: Scene,
        time_scale: float = 1.0,
        preds: list[tuple[str, list[PersonInFrame], tuple[int, int, int] | None]] = []
    ):
        self.draw_scene(scene, time_scale, preds)
        pygame.quit()
