import pygame
import json
import sys
import math

# Load the NDJSON data and separate scenes and tracks
# file_path = 'data_trajnet++/train/real_data/biwi_hotel.ndjson'
file_path = 'data_trajnet++/train/real_data/crowds_students001.ndjson'
scenes = {}
tracks = {}

# Parse the NDJSON file
with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        if 'scene' in data:
            scene = data['scene']
            scenes[scene['id']] = scene
        elif 'track' in data:
            track = data['track']
            pedestrian_id = track['p']
            if pedestrian_id not in tracks:
                tracks[pedestrian_id] = []
            tracks[pedestrian_id].append(track)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((2800, 2000))
pygame.display.set_caption("Trajectory Visualization with Momentum")
clock = pygame.time.Clock()

# Function to draw arrows representing momentum
def draw_arrow(screen, start_pos, momentum, color=(0, 0, 0)):
    end_pos = (start_pos[0] + momentum[0], start_pos[1] + momentum[1])
    pygame.draw.line(screen, color, start_pos, end_pos, 2)  # Line thickness of 2
    angle = math.atan2(momentum[1], momentum[0])
    arrowhead_len = 10
    # Draw two arrowhead lines
    pygame.draw.line(screen, color, end_pos, 
                     (end_pos[0] - arrowhead_len * math.cos(angle - math.pi / 6),
                      end_pos[1] - arrowhead_len * math.sin(angle - math.pi / 6)), 2)
    pygame.draw.line(screen, color, end_pos, 
                     (end_pos[0] - arrowhead_len * math.cos(angle + math.pi / 6),
                      end_pos[1] - arrowhead_len * math.sin(angle + math.pi / 6)), 2)

# Function to draw a frame for a given scene
def draw_scene(scene_id):
    if scene_id not in scenes:
        print(f"Scene {scene_id} not found.")
        return
    
    scene = scenes[scene_id]
    primary_pedestrian = scene['p']
    start_frame, end_frame = scene['s'], scene['e']

    # Filter all tracks for pedestrians within the scene's time range
    all_tracks = {pid: [t for t in tracks[pid] if start_frame <= t['f'] <= end_frame] for pid in tracks}
    
    # Sort frames for each pedestrian to estimate momentum
    for pid, track_data in all_tracks.items():
        track_data.sort(key=lambda t: t['f'])  # Sort by frame number

    # Get a sorted list of unique frames within the scene's range
    all_frames = sorted({t['f'] for pid in all_tracks for t in all_tracks[pid]})

    for frame in all_frames:
        screen.fill((255, 255, 255))  # Clear screen with white background

        # Draw each pedestrian present in the current frame
        for pid, track_data in all_tracks.items():
            current_track = None
            next_track = None

            # Find current and next track points
            for idx, track in enumerate(track_data):
                if track['f'] == frame:
                    current_track = track
                    next_track = track_data[idx + 1] if idx + 1 < len(track_data) else None
                    break
            
            if current_track:
                x, y = current_track['x'], current_track['y']
                pos_x = int((x + 10) * 40)  # Scale and offset for Pygame screen
                pos_y = int((y + 10) * 40)
                
                # Draw pedestrian
                color = (255, 0, 0) if pid == primary_pedestrian else (0, 0, 255)
                pygame.draw.circle(screen, color, (pos_x, pos_y), 5)

                # Calculate momentum vector (approximation using next frame)
                if next_track:
                    next_x, next_y = next_track['x'], next_track['y']
                    momentum_x = (next_x - x) * 40  # Scale momentum for visualization
                    momentum_y = (next_y - y) * 40
                    draw_arrow(screen, (pos_x, pos_y), (momentum_x, momentum_y))

        pygame.display.flip()
        clock.tick(int(scene['fps']))  # Control frame rate

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

# Main function to visualize a scene
def visualize_scene(scene_id):
    try:
        draw_scene(scene_id)
    except KeyboardInterrupt:
        pygame.quit()

# Run the visualization for a specific scene
visualize_scene(1)
pygame.quit()
