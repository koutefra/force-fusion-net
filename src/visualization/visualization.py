import pygame
from pygame import Surface
from entities.vector2d import Point2D
from entities.scene import Scene, Obstacle
from entities.frame import Frame, PersonInFrame
import math
import sys

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

    background_color: tuple[int, int, int] = default_colors["grass_dark"]
    focus_person_color: tuple[int, int, int] = default_colors["red_orange"]
    other_person_color: tuple[int, int, int] = default_colors["sky_blue"]
    velocity_color: tuple[int, int, int] = default_colors["blue"]
    acceleration_color: tuple[int, int, int] = default_colors["red"]
    obstacle_color: tuple[int, int, int] = default_colors["dark_gray"]

    def __init__(
        self, 
        target_screen_size: int = 1000, 
        circle_radius: int = 10, 
        text_size: int = 15,
        vector_line_width: int = 3, 
        arrow_size: int = 10, 
        obstacle_line_width: int = 4,
        arrow_angle: float = math.pi / 6
    ) -> None:
        self.target_screen_size = target_screen_size
        self.circle_radius = circle_radius
        self.text_size = text_size
        self.vector_line_width = vector_line_width
        self.arrow_size = arrow_size
        self.obstacle_line_width = obstacle_line_width
        self.arrow_angle = arrow_angle

    def draw_arrowhead(
        self, 
        screen: Surface, 
        start_pos: Point2D, 
        end_pos: Point2D, 
        color: tuple[int, int, int]
    ) -> None:
        """Draw an arrowhead at the end of a line."""
        angle = math.atan2(end_pos.y - start_pos.y, end_pos.x - start_pos.x)

        left_arrowhead = (
            end_pos.x - self.arrow_size * math.cos(angle + self.arrow_angle),
            end_pos.y - self.arrow_size * math.sin(angle + self.arrow_angle)
        )
        right_arrowhead = (
            end_pos.x - self.arrow_size * math.cos(angle - self.arrow_angle),
            end_pos.y - self.arrow_size * math.sin(angle - self.arrow_angle)
        )

        pygame.draw.line(screen, color, (end_pos.x, end_pos.y), left_arrowhead, 2)
        pygame.draw.line(screen, color, (end_pos.x, end_pos.y), right_arrowhead, 2)

    def draw_labels(
        self, 
        screen: Surface, 
        labels: list[tuple[str, tuple[int, int, int]]], 
        font: pygame.font.Font
    ) -> None:
        """Draw color labels for the visualization."""
        for i, (label, color) in enumerate(labels):
            label_surface = font.render(label, True, color)
            screen.blit(label_surface, (10, 10 + i * 20))

    def draw_person(
        self, 
        screen: Surface, 
        person: PersonInFrame, 
        min_pos: Point2D, 
        spatial_scale: float
    ) -> None:
        """Draw a single person with their velocity and acceleration vectors."""
        scaled_position = (person.position - min_pos) * spatial_scale
        pygame.draw.circle(screen, self.other_person_color, scaled_position.to_tuple(), self.circle_radius)

        # Draw velocity and acceleration vectors
        vectors = [
            (person.velocity * spatial_scale, self.velocity_color), 
            (person.acceleration * spatial_scale, self.acceleration_color)
        ]
        for vector, color in vectors:
            end_pos = scaled_position + vector
            pygame.draw.line(screen, color, scaled_position.to_tuple(), end_pos.to_tuple(), self.vector_line_width)
            self.draw_arrowhead(screen, scaled_position, end_pos, color)

    def draw_obstacles(
        self, 
        screen: Surface, 
        obstacles: list[Obstacle], 
        min_pos: Point2D, 
        spatial_scale: float
    ) -> None:
        """Draw all obstacles as connected lines between each pair of vertices."""
        for obstacle in obstacles:
            scaled_vertices = [(vertex - min_pos) * spatial_scale for vertex in obstacle.vertices]
            for i in range(len(scaled_vertices) - 1):
                start_vertex = scaled_vertices[i].to_tuple()
                end_vertex = scaled_vertices[i + 1].to_tuple()
                pygame.draw.line(screen, self.obstacle_color, start_vertex, end_vertex, width=self.obstacle_line_width)

    def setup_screen(self, scene: Scene, min_pos: Point2D, max_pos: Point2D) -> tuple[Surface, float]:
        """Set up the pygame screen and scaling factor based on scene dimensions."""
        bbox_size = max_pos - min_pos
        spatial_scale = self.target_screen_size / max(bbox_size)
        screen_res = (bbox_size * spatial_scale).to_int_tuple()
        screen = pygame.display.set_mode(screen_res)
        screen.fill(self.background_color)
        pygame.display.flip()
        pygame.display.set_caption(f"Scene {scene.id}; Scale 1:{spatial_scale:.0f}; Res: {screen_res[0]}x{screen_res[1]}")
        return screen, spatial_scale

    def draw_frame(
        self, 
        screen: Surface, 
        scene: Scene, 
        frame: Frame, 
        min_pos: Point2D, 
        spatial_scale: float
    ) -> None:
        """Draw all persons and obstacles for a single frame."""
        for person in frame.persons.values():
            self.draw_person(screen, person, min_pos, spatial_scale)
        self.draw_obstacles(screen, scene.obstacles, min_pos, spatial_scale)

    def draw_scene(
        self, 
        scene: Scene, 
        time_scale: float = 1.0
    ) -> None:
        """Initialize and display the scene frame by frame."""
        pygame.init()
        min_pos, max_pos = scene.bounding_box
        screen, spatial_scale = self.setup_screen(scene, min_pos, max_pos)
        clock = pygame.time.Clock()
        scaled_fps = int(scene.fps * time_scale)
        font = pygame.font.SysFont(None, self.text_size)
        labels = [
            ("Persons", self.other_person_color),
            ("Velocity", self.velocity_color),
            ("Acceleration", self.acceleration_color),
        ]

        for frame in scene.frames.values():
            self.draw_frame(screen, scene, frame, min_pos, spatial_scale)
            self.draw_labels(screen, labels, font)
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
        time_scale: float = 1.0
    ) -> None:
        """Public method to visualize the scene."""
        self.draw_scene(scene, time_scale)
        pygame.quit()
