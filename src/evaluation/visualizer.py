from entities.scene import Scene
import matplotlib.pyplot as plt
from typing import Optional

class Visualizer:
    def __init__():
        pass

    def plot_trajectories(scene: Scene, title: Optional[str] = None, output_file_path: Optional[str] = None) -> None:
        # Initialize plot
        plt.figure(figsize=(10, 10))
        if title:
            plt.title(title)
        plt.grid(True)

        # Colors for trajectories
        colors = plt.cm.get_cmap("tab20", len(scene.frames[list(scene.frames.keys())[0]].persons))

        # Collect and plot trajectories
        for i, (person_id, person_trajectory) in enumerate(scene.frames.to_trajectories().items()):
            trajectory_x = [p.position.x for p in person_trajectory.records.values()]
            trajectory_y = [p.position.y for p in person_trajectory.records.values()]
            plt.plot(trajectory_x, trajectory_y, label=f"Person {person_id}", color=colors(i))

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        # Save the plot if output directory is specified
        if output_file_path:
            plt.savefig(output_file_path)

        plt.show()