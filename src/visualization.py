import pygame
from pygame import Surface
from models.predictor_base import Prediction
from core.vector2d import Position
import math
import sys
from typing import Tuple
from data.dataset import PedestrianDataset

class Visualizer:
    def __init__(self, target_screen_size: int = 1000, circle_size: int = 10, text_size: int = 15,
                 max_vector_magnitude: float = 200):
        self.target_screen_size = target_screen_size
        self.circle_size = circle_size
        self.text_size = text_size
        self.max_vector_magnitude = max_vector_magnitude

    @staticmethod
    def draw_arrowhead(screen: Surface, start_pos: Position, end_pos: Position, color: Tuple[int, int, int]):
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

    def draw_scene(self, scene: PedestrianDataset.Scene, time_scale: float = 1.0, name: str = None,
                   prediction: Prediction = None):
        position_range = scene['max_position'] - scene['min_position']
        position_range_max = max(position_range.x, position_range.y)
        spatial_scale = self.target_screen_size / position_range_max

        # initialize
        pygame.init()
        screen_position_range = (position_range * spatial_scale).to_int_tuple()
        screen = pygame.display.set_mode((screen_position_range[0], screen_position_range[1]))
        screen.fill((255, 255, 255))
        pygame.display.flip()
        clock = pygame.time.Clock()
        scaled_fps = int(scene['fps'] * time_scale)
        font = pygame.font.SysFont(None, self.text_size)
        pygame.display.set_caption(f"Scene {scene['id']},{f' {name}' if name else ''}, Scale 1:{spatial_scale:.0f}, " 
                                   f"{screen_position_range[0]}x{screen_position_range[1]}")

        # iterate through frames
        prev_frame_id = next(iter(scene['positions']))[0]
        for pos_id, pos in scene['positions'].items():
            frame_id, person_id = pos_id

            if frame_id != prev_frame_id:
                pygame.display.flip()
                clock.tick(scaled_fps)
                screen.fill((255, 255, 255))
                prev_frame_id = frame_id

            # draw current position
            scaled_pos = (pos - scene['min_position']) * spatial_scale
            pygame.draw.circle(screen, (255, 0, 0), (scaled_pos.x, scaled_pos.y), self.circle_size)  # red color, circle size 20

            # draw the person_id in the middle of the circle
            text_surface = font.render(str(person_id), True, (0, 0, 0))  # black text
            text_rect = text_surface.get_rect(center=(scaled_pos.x, scaled_pos.y))
            screen.blit(text_surface, text_rect)
            
            # draw the goal position
            last_frame_id = scene['end_frames'][person_id]
            last_pos_id = (last_frame_id, person_id)
            last_pos = scene['positions'][last_pos_id]
            scaled_last_pos = (last_pos - scene['min_position']) * spatial_scale
            pygame.draw.rect(
                screen, (255, 0, 0), 
                pygame.Rect(
                    scaled_last_pos.x - self.circle_size, scaled_last_pos.y - self.circle_size,
                    self.circle_size * 2, self.circle_size * 2
                )
            )

            # draw the person_id in the middle of the rectangle
            text_surface = font.render(str(person_id), True, (0, 0, 0))  # black text
            text_rect = text_surface.get_rect(center=(scaled_last_pos.x, scaled_last_pos.y))
            screen.blit(text_surface, text_rect)

            # if predictions are provided, visualize them
            if prediction:
                # velocity prediction
                if prediction['predicted_velocities'] and pos_id in prediction['predicted_velocities']:
                    velocity = prediction['predicted_velocities'][pos_id]
                    scaled_velocity = (velocity * spatial_scale * time_scale).clamp(self.max_vector_magnitude)
                    velocity_end_pos = Position(x=scaled_pos.x + scaled_velocity.x,
                                                y=scaled_pos.y + scaled_velocity.y)
                    pygame.draw.line(screen, (0, 0, 255), (scaled_pos.x, scaled_pos.y), 
                                    (velocity_end_pos.x, velocity_end_pos.y), 3)
                    self.draw_arrowhead(screen, scaled_pos, velocity_end_pos, (0, 0, 255))

                # force prediction
                if prediction['predicted_forces'] and pos_id in prediction['predicted_forces']:
                    force = prediction['predicted_forces'][pos_id]
                    scaled_force = (force * spatial_scale * (time_scale ** 2)).clamp(self.max_vector_magnitude)
                    force_end_pos = Position(x=scaled_pos.x + scaled_force.x,
                                            y=scaled_pos.y + scaled_force.y)
                    pygame.draw.line(screen, (255, 0, 0), (scaled_pos.x, scaled_pos.y), 
                                    (force_end_pos.x, force_end_pos.y), 3)
                    self.draw_arrowhead(screen, scaled_pos, force_end_pos, (255, 0, 0))

            # handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def visualize(self, scene: PedestrianDataset.Scene, time_scale: float = 1.0, name: str = None,
                  prediction: Prediction = None):
        self.draw_scene(scene, time_scale, name, prediction)
        pygame.quit()
