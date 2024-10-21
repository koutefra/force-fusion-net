import pygame
from pedestrian_dataset import PedestrianDataset
import math


def visualize(scene: PedestrianDataset.Scene, res: tuple[int, int] = (800, 600), 
              padding: int = 50, circle_size: int = 10):
    pygame.init()
    width, height = res
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Scene {scene['id']} Trajectories Visualization")
    clock = pygame.time.Clock()

    def draw_scene():
        primary_pedestrian = scene['p']
        min_x = min(track['x'] for track in scene['tracks'])
        max_x = max(track['x'] for track in scene['tracks'])
        min_y = min(track['y'] for track in scene['tracks'])
        max_y = max(track['y'] for track in scene['tracks'])
        prev_frame_id = None

        for track in scene['tracks']:
            cur_frame_id = track['f']

            if cur_frame_id != prev_frame_id:
                pygame.display.flip()
                clock.tick(int(scene['fps']))
                screen.fill((255, 255, 255))
                prev_frame_id = cur_frame_id

            normed_x = (track['x'] - min_x) / (max_x - min_x) # x between 0-1
            normed_y = (track['y'] - min_y) / (max_y - min_y) # y between 0-1
            pos_x = round(normed_x * (width - 2 * padding) + padding)
            pos_y = round(normed_y * (height - 2 * padding) + padding)
            
            color = (255, 0, 0) if track['p'] == primary_pedestrian else (0, 0, 255)
            pygame.draw.circle(screen, color, (pos_x, pos_y), circle_size)

            # Handle Pygame events
            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         pygame.quit()
            #         sys.exit()

    draw_scene()

    pygame.quit()
