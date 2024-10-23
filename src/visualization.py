import pygame
import colorsys
from pedestrian_dataset import PedestrianDataset
from predictor_base import Prediction
import math
import sys
from typing import List, Tuple

class Visualizer:
    def __init__(self, res: tuple[int, int] = (800, 600), padding: int = 50, circle_size: int = 20):
        self.res = res
        self.padding = padding
        self.circle_size = circle_size

        pygame.init()
        self.width, self.height = self.res
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill((255, 255, 255))
        pygame.display.flip()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 25)

    @staticmethod
    def generate_distinct_colors(num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate a list of visually distinct RGB colors."""
        colors = []
        for i in range(num_colors):
            # Use colorsys to get distinct hues evenly spaced in the color wheel
            hue = i / num_colors  # even spacing on the hue spectrum
            saturation = 0.9  # high saturation for vivid colors
            brightness = 0.9  # high brightness for better contrast

            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, brightness)
            # Convert float RGB (0.0 - 1.0) to integer (0 - 255)
            rgb = tuple(int(c * 255) for c in rgb)
            colors.append(rgb)
        return colors

    @staticmethod
    def norm(pos: PedestrianDataset.Pos, min_x: float, max_x: float, min_y: float, max_y: float) -> PedestrianDataset.Pos:
        normed = PedestrianDataset.Pos(x=-1, y=-1)
        for coord, (min_val, max_val) in [('x', (min_x, max_x)), ('y', (min_y, max_y))]:
            # value between 0-1
            normed[coord] = (pos[coord] - min_val) / (max_val - min_val)
        return normed

    def align_pos(self, normed_pos: PedestrianDataset.Pos) -> PedestrianDataset.Pos:
        pos_x = round(normed_pos['x'] * (self.width - 2 * self.padding) + self.padding)
        pos_y = round(normed_pos['y'] * (self.height - 2 * self.padding) + self.padding)
        return PedestrianDataset.Pos(x=pos_x, y=pos_y)

    def draw_arrowhead(self, start_pos: PedestrianDataset.Pos, end_pos: PedestrianDataset.Pos, color: Tuple[int, int, int]):
        """Draw an arrowhead at the end of a line"""
        angle = math.atan2(end_pos['y'] - start_pos['y'], end_pos['x'] - start_pos['x'])
        arrow_size = 10  # Arrowhead size
        arrow_angle = math.pi / 6  # Angle of the arrowhead lines

        left_arrowhead = (
            end_pos['x'] - arrow_size * math.cos(angle + arrow_angle),
            end_pos['y'] - arrow_size * math.sin(angle + arrow_angle)
        )
        right_arrowhead = (
            end_pos['x'] - arrow_size * math.cos(angle - arrow_angle),
            end_pos['y'] - arrow_size * math.sin(angle - arrow_angle)
        )

        pygame.draw.line(self.screen, color, (end_pos['x'], end_pos['y']), left_arrowhead, 3)
        pygame.draw.line(self.screen, color, (end_pos['x'], end_pos['y']), right_arrowhead, 3)

    def draw_scene(self, scene: PedestrianDataset.Scene, prediction: Prediction = None):
        pygame.display.set_caption(f"Scene {scene['id']} Trajectories Visualization")
        min_x = min(track['x'] for track in scene['positions'].values())
        max_x = max(track['x'] for track in scene['positions'].values())
        min_y = min(track['y'] for track in scene['positions'].values())
        max_y = max(track['y'] for track in scene['positions'].values())

        num_persons = len(scene['pedestrian_ids'])
        colors_list = self.generate_distinct_colors(num_persons)
        colors = {pid: colors_list[idx] for idx, pid in enumerate(scene['pedestrian_ids'])}

        prev_frame_id = next(iter(scene['positions']))[0]
        for pos_id, pos in scene['positions'].items():
            frame_id, person_id = pos_id
            cur_color = colors[person_id]

            if frame_id != prev_frame_id:
                pygame.display.flip()
                self.clock.tick(int(scene['fps']))
                self.screen.fill((255, 255, 255))
                prev_frame_id = frame_id

            # draw current position
            normed_cur_pos = self.norm(pos, min_x, max_x, min_y, max_y)
            aligned_cur_pos = self.align_pos(normed_cur_pos)
            pygame.draw.circle(self.screen, cur_color, (aligned_cur_pos['x'], aligned_cur_pos['y']), self.circle_size)

            # draw the person_id in the middle of the circle
            text_surface = self.font.render(str(person_id), True, (0, 0, 0))  # Black text
            text_rect = text_surface.get_rect(center=(aligned_cur_pos['x'], aligned_cur_pos['y']))
            self.screen.blit(text_surface, text_rect)
            
            # draw the goal position
            last_frame_id = scene['end_frames'][person_id]
            last_pos_id = (last_frame_id, person_id)
            last_pos = scene['positions'][last_pos_id]
            normed_last_pos = self.norm(last_pos, min_x, max_x, min_y, max_y)
            aligned_last_pos = self.align_pos(normed_last_pos)
            pygame.draw.rect(
                self.screen, cur_color, 
                pygame.Rect(
                    aligned_last_pos['x'] - self.circle_size, aligned_last_pos['y'] - self.circle_size,
                    self.circle_size * 2, self.circle_size * 2
                )
            )

            # draw the person_id in the middle of the rectangle
            text_surface = self.font.render(str(person_id), True, (0, 0, 0))  # Black text
            text_rect = text_surface.get_rect(center=(aligned_last_pos['x'], aligned_last_pos['y']))
            self.screen.blit(text_surface, text_rect)

            # If predictions are provided, draw the velocity and force arrows
            if prediction and (frame_id, person_id) in prediction['preds']:
                pred = prediction['preds'][(frame_id, person_id)]

                # Draw velocity arrow
                velocity = pred[0]  # (vx, vy)
                velocity_end_pos = PedestrianDataset.Pos(
                    x=aligned_cur_pos['x'] + int(velocity['vx'] * 20),  # Scale velocity for visualization
                    y=aligned_cur_pos['y'] + int(velocity['vy'] * 20)
                )
                pygame.draw.line(self.screen, (0, 0, 255), (aligned_cur_pos['x'], aligned_cur_pos['y']), 
                                (velocity_end_pos['x'], velocity_end_pos['y']), 3)
                # Add arrowhead for velocity
                self.draw_arrowhead(aligned_cur_pos, velocity_end_pos, (0, 0, 255))

                # Draw force arrow
                force = pred[1]  # (fx, fy)
                force_end_pos = PedestrianDataset.Pos(
                    x=aligned_cur_pos['x'] + int(force['fx'] * 20),  # Scale force for visualization
                    y=aligned_cur_pos['y'] + int(force['fy'] * 20)
                )
                pygame.draw.line(self.screen, (255, 0, 0), (aligned_cur_pos['x'], aligned_cur_pos['y']), 
                                (force_end_pos['x'], force_end_pos['y']), 3)
                # Add arrowhead for force
                self.draw_arrowhead(aligned_cur_pos, force_end_pos, (255, 0, 0))

            # handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def visualize(self, scene: PedestrianDataset.Scene, predictions: List[Prediction] = None):
        self.draw_scene(scene, predictions)
        pygame.quit()
