# ---- add near top of file (helpers) ----
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from entities.scene import Scene
from entities.vector2d import Point2D
from entities.vector2d import closest_point_on_line  # add this import
import matplotlib.pyplot as plt


def _finite_any(arr) -> bool:
    if arr is None: return False
    a = np.asarray(arr)
    return np.isfinite(a).any()


def _closest_agent_and_obstacle_distances_per_frame(
    scene: Scene, person_id: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (frames, min_dist_to_agent, min_dist_to_obstacle) for the given person.
    Obstacle is min distance to any line segment in the frame.
    """
    ts, d_agents, d_obs = [], [], []
    for t, fr in sorted(scene.frames.items()):
        me = fr.persons.get(person_id)
        if me is None:
            continue

        # closest agent
        dmin_a = np.inf
        for oid, op in fr.persons.items():
            if oid == person_id: continue
            dx = me.position.x - op.position.x
            dy = me.position.y - op.position.y
            d = (dx*dx + dy*dy) ** 0.5
            if d < dmin_a: dmin_a = d
        if not np.isfinite(dmin_a): dmin_a = np.nan

        # closest obstacle (line segments)
        dmin_o = np.inf
        for line in fr.obstacles:
            p_closest = closest_point_on_line(me.position, line.p1, line.p2)
            dx = me.position.x - p_closest.x
            dy = me.position.y - p_closest.y
            d = (dx*dx + dy*dy) ** 0.5
            if d < dmin_o: dmin_o = d
        if not np.isfinite(dmin_o): dmin_o = np.nan

        ts.append(t); d_agents.append(dmin_a); d_obs.append(dmin_o)

    return np.asarray(ts, dtype=int), np.asarray(d_agents, float), np.asarray(d_obs, float)


def _mag_from_scene(
    scene: Scene, person_id: int, which: str = "total"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fallback magnitude series computed directly from the scene if not present in alignment dict.
    which: 'total' | 'desired' | 'rep_obs' | 'rep_agents'
    Returns (frames, |a|) with NaNs where unavailable.
    """
    ts, mags = [], []
    for t, fr in sorted(scene.frames.items()):
        p = fr.persons.get(person_id)
        if p is None:
            continue
        a = None
        if which == "total":
            a = p.acceleration
        else:
            if getattr(p, "forces", None) is not None:
                if which == "desired":
                    a = p.forces.desired
                elif which == "rep_obs":
                    a = p.forces.repulsive_obs
                elif which == "rep_agents":
                    a = p.forces.repulsive_agents
        ts.append(t)
        mags.append((a.magnitude() if a is not None else np.nan))
    return np.asarray(ts, dtype=int), np.asarray(mags, dtype=float)


def _closest_agent_distance_per_frame(scene: Scene, person_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (frames, min_distance_to_any_other_agent) for a given person."""
    frames = []
    dmins = []
    for t, fr in sorted(scene.frames.items()):
        me = fr.persons.get(person_id)
        if me is None:
            continue
        if not fr.persons:
            continue
        dmin = np.inf
        for oid, op in fr.persons.items():
            if oid == person_id: continue
            dx = me.position.x - op.position.x
            dy = me.position.y - op.position.y
            d = (dx*dx + dy*dy) ** 0.5
            if d < dmin: dmin = d
        frames.append(t)
        dmins.append(dmin if np.isfinite(dmin) else np.nan)
    return np.asarray(frames, dtype=int), np.asarray(dmins, dtype=float)


class Visualizer:
    def __init__():
        pass

    @staticmethod
    def plot_trajectories(
        scene: Scene,
        title: Optional[str] = None,
        output_file_path: Optional[str] = None,
        show_obstacles: bool = True,
        show_bounding_box: bool = True,
    ) -> None:
        """Plot all pedestrian trajectories with optional obstacles and bounding box."""
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        if title:
            plt.title(title)

        # --- Grid & aesthetics ---
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_aspect("equal", adjustable="box")

        # --- Plot trajectories ---
        first_frame = next(iter(scene.frames.values()))
        persons = first_frame.persons
        colors = plt.cm.get_cmap("tab20", len(persons))

        for i, (pid, traj) in enumerate(scene.frames.to_trajectories().items()):
            xs = [p.position.x for p in traj.records.values()]
            ys = [p.position.y for p in traj.records.values()]
            ax.plot(xs, ys, lw=1.8, color=colors(i), label=f"Person {pid}")

        # --- Plot obstacles ---
        if show_obstacles:
            for frame in scene.frames.values():
                if not hasattr(frame, "obstacles") or not frame.obstacles:
                    continue
                for obs in frame.obstacles:
                    xs = [obs.p1.x, obs.p2.x]
                    ys = [obs.p1.y, obs.p2.y]
                    ax.plot(xs, ys, color="black", lw=2.0, alpha=0.8)
                break  # obstacles are static → draw once

        # --- Bounding box ---
        if show_bounding_box and hasattr(scene, "bounding_box"):
            (p_min, p_max) = scene.bounding_box
            rect_x = [p_min.x, p_max.x, p_max.x, p_min.x, p_min.x]
            rect_y = [p_min.y, p_min.y, p_max.y, p_max.y, p_min.y]
            ax.plot(rect_x, rect_y, "k--", lw=1.2, label="Bounding box")

        # --- Layout & output ---
        ax.set_xticks([]); ax.set_yticks([])
        ax.invert_yaxis()

        plt.tight_layout()

        if output_file_path:
            plt.savefig(output_file_path, dpi=300)
            print(f"[INFO] Trajectories saved to {output_file_path}")

        plt.show()

    @staticmethod
    def plot_collision_vs_threshold_multi(
        data_dict: dict[str, dict[str, np.ndarray]],
        key: str,
        title: str | None = None,
        out_path: str | None = None,
        dpi: int = 220,
    ) -> None:
        """
        Plot multi-model comparison for collision-vs-threshold curves.

        Args:
            data_dict: mapping model_name → result dict from Evaluator.evaluate_collision_vs_threshold()
            key: 'agent_collisions' or 'obstacle_collisions'
            title: figure title
            out_path: optional save path
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5), dpi=dpi)
        palette = ["#1976d2", "#e53935", "#43a047", "#8e24aa", "#fb8c00", "#00897b", "#7b1fa2", "#5d4037"]

        for i, (name, data) in enumerate(data_dict.items()):
            thresholds = data["thresholds"]
            values = data[key]
            plt.plot(thresholds, values, "o-", lw=2.0, color=palette[i % len(palette)], label=name)

        plt.xlabel("Collision threshold [m]")
        plt.ylabel("Total collisions")
        plt.title(title or f"{key.replace('_', ' ').title()}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
            plt.close()
            print(f"[Visualizer] saved {key} plot → {out_path}")
        else:
            plt.show()
