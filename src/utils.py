from vector2d import Position, Velocity, Force
from typing import List, Tuple
import colorsys

def norm(val: float, min_val: float, max_val: float) -> float:
    return (val - min_val) / (max_val - min_val)

def norm_position(pos: Position, pos_min: Position, pos_max: Position) -> Position:
    return Position(x=norm(pos['x'], pos_min['x'], pos_max['x']), 
                    y=norm(pos['y'], pos_min['y'], pos_max['y']))

def scale_velocity(velocity: Velocity, spatial_scale: float, time_scale: float) -> Velocity:
    """Convert real-world velocity (m/s) to simulation velocity (pixels/frame)."""
    scaled_vx = velocity['vx'] * spatial_scale * time_scale
    scaled_vy = velocity['vy'] * spatial_scale * time_scale
    return Velocity(vx=scaled_vx, vy=scaled_vy)

def scale_force(force: Force, spatial_scale: float, time_scale: float) -> Force:
    """Convert real-world force to simulation force in pixels based on spatial and time scaling."""
    scaled_fx = force['fx'] * spatial_scale * (time_scale ** 2)
    scaled_fy = force['fy'] * spatial_scale * (time_scale ** 2)
    return Force(fx=scaled_fx, fy=scaled_fy)

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