# ---- add near top of file (helpers) ----
import numpy as np
from typing import Optional, Dict
from pathlib import Path
from entities.scene import Scene
from entities.vector2d import Point2D
from evaluation.align import compute_force_alignment  # assumed available
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

    # ================== NEW 2×5 comparison figure ==================
    @staticmethod
    def plot_force_comparison_2x5(
        models: Dict[str, Scene],
        person_id: int,
        ref_point: Point2D = Point2D(0.0, 0.0),
        out_path: Optional[str] = None,
        show_components: bool = True,
        dpi: int = 220,            # high DPI so you can zoom in cleanly
        figsize: tuple = (22, 9),  # big canvas for publication
    ) -> None:
        """
        2x5 grid:
          (1,1) Trajectory                (2,1) Closest-agent distance
          (1,2) Total cos                 (2,2) Total |a|
          (1,3) Desired cos               (2,3) Desired |a|
          (1,4) RepObs cos                (2,4) RepObs |a|
          (1,5) RepInt cos                (2,5) RepInt |a|
        Overlays multiple models per axis (color per model).
        """
        import matplotlib.pyplot as plt

        if not models:
            raise ValueError("No models provided.")

        names = list(models.keys())
        palette = ["#1976d2", "#e53935", "#43a047", "#fb8c00", "#8e24aa", "#00897b", "#7b1fa2", "#5d4037"]
        colors = {name: palette[i % len(palette)] for i, name in enumerate(names)}

        # Use any scene for bounds/obstacles
        any_scene: Scene = next(iter(models.values()))
        bl, tr = any_scene.bounding_box
        xmin, xmax, ymin, ymax = bl.x, tr.x, bl.y, tr.y

        # Alignment series per model
        series = {}
        have_des = have_ra = have_ro = False
        for name, scene in models.items():
            d = compute_force_alignment(scene, ref_point, person_ids=[person_id]).get(person_id)
            if d is None or len(d["t"]) == 0:
                raise ValueError(f"[{name}] person_id={person_id} missing or empty.")
            series[name] = d
            have_des |= _finite_any(d.get("cos_desired"))
            have_ra  |= _finite_any(d.get("cos_rep_agents"))  # internal / interaction
            have_ro  |= _finite_any(d.get("cos_rep_obs"))

        # Precompute closest-agent distance per model
        closest = {name: _closest_agent_distance_per_frame(scene, person_id) for name, scene in models.items()}

        # Prepare figure: 2 rows × 5 cols
        fig, axs = plt.subplots(2, 5, figsize=figsize, dpi=dpi,
                                gridspec_kw=dict(wspace=0.30, hspace=0.35))

        # ========== (1,1) Trajectory ==========
        ax = axs[0, 0]
        ax.set_title(f"Trajectory — pid {person_id}")
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.2)
        ax.axvline(ref_point.x, color="k", lw=1.2, ls="--", alpha=0.7)

        # obstacles from first frame of any_scene
        first_frame = next(iter(any_scene.frames.values()))
        for o in first_frame.obstacles:
            ax.plot([o.p1.x, o.p2.x], [o.p1.y, o.p2.y], color="0.5", lw=2, alpha=0.6)

        for name, scene in models.items():
            xs, ys = [], []
            for t, fr in sorted(scene.frames.items()):
                p = fr.persons.get(person_id)
                if p:
                    xs.append(p.position.x); ys.append(p.position.y)
            if xs:
                ax.plot(xs, ys, "-", lw=2.0, color=colors[name], label=name)
                ax.plot(xs[0], ys[0], "o", ms=5, color=colors[name])

        # ========== (2,1) Closest distances (agents & obstacles) ==========
        ax = axs[1, 0]
        ax.set_title("Closest distance")
        ax.set_xlabel("frame"); ax.set_ylabel("min distance [m]")
        ax.grid(True, alpha=0.3)

        # compute once from the first (reference) scene
        ref_name, ref_scene = next(iter(models.items()))
        tt_ref, d_agents, d_obs = _closest_agent_and_obstacle_distances_per_frame(ref_scene, person_id)

        # two curves: agents + obstacles
        h1 = ax.plot(tt_ref, d_agents, "-", lw=1.8, color="0.15", label="closest agent")
        h2 = ax.plot(tt_ref, d_obs,   "-", lw=1.8, color="0.45", label="closest obstacle")

        # emphasize reference crossing (same index we use in the cos/mag panels)
        # pick the first model's alignment dict for x_reach_idx
        d_first = next(iter(series.values()))
        xr = d_first.get("x_reach_idx")
        if xr is not None and np.size(xr) > 0:
            xr0 = int(xr[0])
            if 0 <= xr0 < len(tt_ref):
                ax.axvline(tt_ref[xr0], color="k", lw=1.2, ls="--", alpha=0.6)

        ax.legend(loc="best", fontsize=9)

        # Helper to draw cos/mag columns
        def _cos_mag(col_idx: int, title: str, ycos_key: str, ymag_key: str, which_mag: str):
            axc = axs[0, col_idx]
            axm = axs[1, col_idx]
            axc.set_title(f"{title} — cos")
            axc.set_xlabel("frame"); axc.set_ylabel("cos θ"); axc.set_ylim(-1.05, 1.05); axc.grid(True, alpha=0.3)
            axm.set_title(f"{title} — |a|")
            axm.set_xlabel("frame"); axm.set_ylabel("|a|"); axm.grid(True, alpha=0.3)

            any_line_c = any_line_m = False
            for name, d in series.items():
                t = d["t"]

                # cosine straight from alignment dict
                yc = d.get(ycos_key)
                if _finite_any(yc):
                    axc.plot(t, yc, "-", lw=1.6, color=colors[name], label=name); any_line_c = True

                # magnitude: try alignment dict key (with backward-compat), else compute from scene
                ym = d.get(ymag_key)
                if ym is None and which_mag == "total":
                    ym = d.get("amag")  # old key for total

                if not _finite_any(ym):
                    # compute from the scene if missing/empty
                    tt_f, ym_f = _mag_from_scene(models[name], person_id, which=which_mag)
                    # align to the same t if necessary (assumes same frames across models; if not, we plot as-is)
                    if len(tt_f) and (len(t) == len(tt_f)) and np.all(tt_f == t):
                        ym = ym_f
                    else:
                        # different timelines: just plot fallback on its own t
                        axm.plot(tt_f, ym_f, "-", lw=1.6, color=colors[name], label=name); any_line_m = True
                        ym = None  # prevent double plot below

                if _finite_any(ym):
                    axm.plot(t, ym, "-", lw=1.6, color=colors[name], label=name); any_line_m = True

                # reference crossing marker
                xr = d.get("x_reach_idx")
                if xr is not None and np.size(xr) > 0:
                    xr0 = int(xr[0])
                    if 0 <= xr0 < len(t):
                        axc.axvline(t[xr0], color=colors[name], lw=1.0, ls="--", alpha=0.5)
                        axm.axvline(t[xr0], color=colors[name], lw=1.0, ls="--", alpha=0.5)

            if any_line_c:
                h, l = axc.get_legend_handles_labels()
                if h: axc.legend(loc="best", fontsize=9)
            if any_line_m:
                h, l = axm.get_legend_handles_labels()
                if h: axm.legend(loc="best", fontsize=9)

        # (1,2)/(2,2): Total
        _cos_mag(col_idx=1, title="Total",
                ycos_key="cos_total", ymag_key="amag_total", which_mag="total")

        # (1,3)/(2,3): Desired (optional)
        if show_components and have_des:
            _cos_mag(col_idx=2, title="Desired",
                    ycos_key="cos_desired", ymag_key="amag_desired", which_mag="desired")

        # (1,4)/(2,4): Repulsive obstacles (optional)
        if show_components and have_ro:
            _cos_mag(col_idx=3, title="Repulsive (obs)",
                    ycos_key="cos_rep_obs", ymag_key="amag_rep_obs", which_mag="rep_obs")

        # (1,5)/(2,5): Repulsive interaction/agents (optional)
        if show_components and have_ra:
            _cos_mag(col_idx=4, title="Repulsive (int)",
                    ycos_key="cos_rep_agents", ymag_key="amag_rep_agents", which_mag="rep_agents")

        # If any optional block is missing, its axes will be empty; that's ok visually.

        plt.tight_layout()
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[Visualizer] saved 2x5 comparison: {out_path}")
        else:
            plt.show()

    @staticmethod
    def plot_force_comparison_batch_2x5(
        models: Dict[str, Scene],
        person_id: Optional[int] = None,
        ref_point: Point2D = Point2D(0.0, 0.0),
        out_dir: str = "results/force_comparison_2x5",
        show_components: bool = True,
        dpi: int = 220,
        figsize: tuple = (22, 9),
    ) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        if person_id is None:
            common = None
            for sc in models.values():
                pids = sc.get_all_person_ids()
                common = pids if common is None else (common & pids)
            if not common:
                raise ValueError("No common person IDs across provided models.")
            pids = sorted(common)
        else:
            pids = [person_id]

        for pid in pids:
            out_path = str(Path(out_dir) / f"pid_{pid:03d}.png")
            Visualizer.plot_force_comparison_2x5(
                models=models,
                person_id=pid,
                ref_point=ref_point,
                out_path=out_path,
                show_components=show_components,
                dpi=dpi,
                figsize=figsize,
            )

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
