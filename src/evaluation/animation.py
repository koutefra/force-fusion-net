import pygame
import re
import os
import shutil
from pygame import Surface
import imageio
import imageio.v3 as iio  # Newer `imageio.v3` API is faster and more efficient
from entities.vector2d import Point2D
from entities.scene import Scene
from entities.person import Person
from entities.frame import Frame
from entities.obstacle import LineObstacle
import math
import sys
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import uuid

class Animation:
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
        target_screen_size: int = 1200, 
        font_size: int = 24,
        circle_radius: int = 10, 
        text_size: int = 20,
        vector_line_width: int = 2, 
        arrow_size: int = 8, 
        obstacle_line_width: int = 5,
        arrow_angle: float = math.pi / 6,
        output_dir: Optional[str] = "animations"
    ):
        self.target_screen_size = target_screen_size
        self.font_size = font_size
        self.circle_radius = circle_radius
        self.text_size = text_size
        self.vector_line_width = vector_line_width
        self.arrow_size = arrow_size
        self.obstacle_line_width = obstacle_line_width
        self.arrow_angle = arrow_angle
        self.output_dir = output_dir

        if self.output_dir:
            random_suffix = str(uuid.uuid4())
            self.output_dir_frames = os.path.join(self.output_dir, f'frames_{random_suffix}')
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir_frames)

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
        person: Person, 
        color: tuple[int, int, int],
        min_pos: Point2D, 
        spatial_scale: float,
        draw_id: bool = False
    ) -> None:
        """Draw a single person with their velocity and acceleration vectors."""
        scaled_position = (person.position - min_pos) * spatial_scale
        pygame.draw.circle(screen, color, scaled_position.to_tuple(), self.circle_radius)

        # Draw the person's ID in the middle of the circle
        if draw_id:
            font = pygame.font.Font(None, self.font_size)
            id_surface = font.render(str(person.id), True, (0, 0, 0))  # Render ID as black text
            id_rect = id_surface.get_rect(center=scaled_position.to_tuple())
            screen.blit(id_surface, id_rect)

        # Draw the goal as a rectangle if it exists
        if person.goal:
            scaled_goal_position = (person.goal - min_pos) * spatial_scale
            pygame.draw.rect(
                screen,
                color,
                pygame.Rect(
                    scaled_goal_position.x - self.circle_radius / 2,
                    scaled_goal_position.y - self.circle_radius / 2,
                    self.circle_radius,
                    self.circle_radius
                )
            )

        vectors = [
            (person.velocity * spatial_scale if person.velocity else None, self.velocity_color),
            (person.acceleration * spatial_scale if person.acceleration else None, self.acceleration_color)
        ]
        vectors = [(vector, color) for vector, color in vectors if vector is not None]

        # Draw valid vectors
        for vector, color in vectors:
            end_pos = scaled_position + vector
            pygame.draw.line(
                screen, 
                color, 
                scaled_position.to_tuple(), 
                end_pos.to_tuple(), 
                self.vector_line_width
            )
            self.draw_arrowhead(screen, scaled_position, end_pos, color)

    def draw_obstacles(
        self, 
        screen: Surface, 
        obstacles: list[LineObstacle], 
        min_pos: Point2D, 
        spatial_scale: float
    ) -> None:
        """Draw all obstacles as connected lines between each pair of vertices."""
        for obstacle in obstacles:
            scaled_start_point = (obstacle.p1 - min_pos) * spatial_scale
            scaled_end_point = (obstacle.p2 - min_pos) * spatial_scale
            pygame.draw.line(
                screen, 
                self.obstacle_color, 
                scaled_start_point, 
                scaled_end_point, 
                width=self.obstacle_line_width
            )

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
        frame_number: int,
        frame: Frame, 
        min_pos: Point2D, 
        spatial_scale: float,
        draw_person_ids: bool = False,
        person_ids: Optional[list[int]] = None
    ) -> None:
        """Draw all persons and obstacles for a single frame."""
        for person_id, person in frame.persons.items():
            color = self.focus_person_color if person_ids and person_id in person_ids else self.other_person_color
            self.draw_person(screen, person, color, min_pos, spatial_scale, draw_id=draw_person_ids)
        self.draw_obstacles(screen, frame.obstacles, min_pos, spatial_scale)

        if self.output_dir:
            pygame.image.save(screen, os.path.join(self.output_dir_frames, f"frame_{frame_number}.png"))

    def draw_scene(
        self, 
        scene: Scene, 
        draw_person_ids: bool = False,
        person_ids: Optional[list[int]] = None,
        time_scale: float = 1.0
    ) -> None:
        """Initialize and display the scene frame by frame."""
        pygame.init()
        min_pos, max_pos = scene.bounding_box
        screen, spatial_scale = self.setup_screen(scene, min_pos, max_pos)
        clock = pygame.time.Clock()
        scaled_fps = float(scene.fps * time_scale)

        for frame_number, frame in scene.frames.items():
            self.draw_frame(screen, frame_number, frame, min_pos, spatial_scale, draw_person_ids=draw_person_ids, person_ids=person_ids)
            pygame.display.flip()
            clock.tick(scaled_fps)
            screen.fill(self.background_color)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def _create_mp4(self, scene_id: str, duration: float = 0.1, desc: Optional[str] = None) -> None:
        # Numerical sort using regex
        def numerical_sort(value: str):
            filename = os.path.basename(value)  # Extract the filename
            match = re.search(r'(\d+)', filename)  # Find numbers in the filename
            return int(match.group()) if match else float('inf')  # Use infinity if no number is found

        # List and sort frame files
        frame_files = sorted(
            [os.path.join(self.output_dir_frames, f) for f in os.listdir(self.output_dir_frames) if f.endswith('.png')],
            key=numerical_sort
        )

        # MP4 filename
        mp4_filename = os.path.join(
            self.output_dir, f"scene_{scene_id}" + (f"_{desc}" if desc else "") + "_animation.mp4"
        )

        # Write frames to video
        with imageio.get_writer(mp4_filename, fps=1/duration) as writer:
            with ThreadPoolExecutor() as executor:
                for frame in executor.map(iio.imread, frame_files):
                    writer.append_data(frame)

        # Clean up frames directory
        shutil.rmtree(self.output_dir_frames)
        print(f"MP4 saved as {mp4_filename}")

    def create(
        self, 
        scene: Scene, 
        desc: Optional[str] = None,
        draw_person_ids: bool = False,
        person_ids: Optional[list[int]] = None,
        time_scale: float = 1.0
    ) -> None:
        self.draw_scene(scene, draw_person_ids=draw_person_ids, person_ids=person_ids, time_scale=time_scale)
        pygame.quit()
        if self.output_dir:
            duration = 1/(scene.fps * time_scale)
            self._create_mp4(scene.id, duration=duration, desc=desc)