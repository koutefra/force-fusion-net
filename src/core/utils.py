from typing import List, Tuple
import colorsys

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