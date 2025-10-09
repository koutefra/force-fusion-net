# evaluation/angle_animator.py
from __future__ import annotations
import os
from typing import Optional, Dict, Tuple, List
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

from entities.scene import Scene
from entities.vector2d import Point2D
from evaluation.align import compute_force_alignment


# --------- small helpers ---------
def _scene_bounds(scene: Scene, pad_ratio: float = 0.02) -> Tuple[float, float, float, float]:
    bl, tr = scene.bounding_box
    pad = pad_ratio * max(tr.x - bl.x, tr.y - bl.y)
    return (bl.x - pad, tr.x + pad, bl.y - pad, tr.y + pad)

def _auto_arrow_scale(xs: np.ndarray, ys: np.ndarray, ax: np.ndarray, ay: np.ndarray,
                      bbox_span: float, base_ratio: float = 0.12, user_scale: float = 1.0) -> float:
    norms = np.hypot(ax, ay)
    ref = np.nanpercentile(norms[norms > 0], 90) if np.any(norms > 0) else 1.0
    return (base_ratio * bbox_span) / max(ref, 1e-9) * user_scale

def _model_colors(names: List[str]) -> Dict[str, str]:
    # consistent, high-contrast palette for 1–6 models
    palette = ["#1976d2", "#e53935", "#43a047", "#fb8c00", "#8e24aa", "#00897b"]
    return {name: palette[i % len(palette)] for i, name in enumerate(names)}

def _has_components(data: dict) -> bool:
    # from compute_force_alignment() arrays presence
    def finite_any(x): return (x is not None) and np.isfinite(x).any()
    return any(finite_any(data.get(k)) for k in ["cos_desired", "cos_rep_agents", "cos_rep_obs",
                                                 "amag_desired", "amag_rep_agents", "amag_rep_obs"])


# --------- main class ---------
class AngleCosineAnimator:
    """
    Video-only renderer (no slider) + static comparison grid across models.
    """

    def __init__(self, scene: Scene, ref_point: Point2D = Point2D(0.0, 0.0)):
        self.scene = scene
        self.ref = ref_point
        self.frames = dict(sorted(scene.frames.items()))
        self.fnums = np.array(list(self.frames.keys()), dtype=int)

    # ---------- VIDEO (single model) ----------
    def render_video(
        self,
        person_id: int,
        save_path: str,
        show_components: bool = True,
        fps: int = 12,
        dpi: int = 110,
        arrow_scale: float = 1.2,
    ) -> None:
        """
        Render MP4 showing: left = scene with big total acceleration arrow,
        right = cos(total & components if present) vs time with moving cursor.
        """
        data = compute_force_alignment(self.scene, self.ref, person_ids=[person_id]).get(person_id)
        if data is None:
            raise ValueError(f"Person {person_id} not found in scene.")

        ts = data["t"]
        cos_total = data["cos_total"]
        cos_des   = data.get("cos_desired")
        cos_ra    = data.get("cos_rep_agents")
        cos_ro    = data.get("cos_rep_obs")
        x_reach_idx = int(data["x_reach_idx"][0]) if data["x_reach_idx"].size else -1

        # focused trajectory + accelerations per frame
        xs, ys, ax_all, ay_all = [], [], [], []
        for t in ts:
            p = self.frames[t].persons.get(person_id)
            xs.append(p.position.x)
            ys.append(p.position.y)
            if p.acceleration is not None:
                ax_all.append(p.acceleration.x); ay_all.append(p.acceleration.y)
            else:
                ax_all.append(0.0); ay_all.append(0.0)
        xs, ys = np.asarray(xs), np.asarray(ys)
        ax_all, ay_all = np.asarray(ax_all), np.asarray(ay_all)

        # faint crowd per frame (context)
        all_pts_per_frame = []
        for t in ts:
            fr = self.frames[t]
            if fr.persons:
                pts = np.array([[pp.position.x, pp.position.y] for pp in fr.persons.values()], dtype=float)
            else:
                pts = np.zeros((0, 2), dtype=float)
            all_pts_per_frame.append(pts)

        # scene bounds & arrow scaling
        xmin, xmax, ymin, ymax = _scene_bounds(self.scene)
        span = max(xmax - xmin, ymax - ymin)
        viz_scale = _auto_arrow_scale(xs, ys, ax_all, ay_all, span, base_ratio=0.12, user_scale=arrow_scale)

        # figure layout
        fig = plt.figure(figsize=(10, 5), dpi=dpi)
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)
        ax_scene = fig.add_subplot(gs[0, 0])
        ax_plot  = fig.add_subplot(gs[0, 1])

        # left: scene
        ax_scene.set_xlim(xmin, xmax); ax_scene.set_ylim(ymin, ymax)
        ax_scene.set_aspect("equal", adjustable="box")
        ax_scene.set_title(f"{self.scene.id} — Person {person_id}")
        ax_scene.grid(True, alpha=0.2)

        # obstacles (draw once from first frame)
        first = next(iter(self.frames.values()))
        for o in first.obstacles:
            ax_scene.plot([o.p1.x, o.p2.x], [o.p1.y, o.p2.y], color="0.4", lw=2, alpha=0.6)

        # ref vertical
        ax_scene.axvline(self.ref.x, color="k", lw=1.5, ls="--", alpha=0.7)

        # all agents (faint)
        all_agents_scatter = ax_scene.scatter([], [], s=8, c="tab:cyan", alpha=0.35, zorder=1)

        # focused path + point
        (traj_line,) = ax_scene.plot(xs, ys, "-", lw=1.5, alpha=0.35, color="tab:blue", zorder=2)
        (cur_point,) = ax_scene.plot([xs[0]], [ys[0]], "o", ms=8, color="tab:red", zorder=4,
                                     path_effects=[pe.Stroke(linewidth=3, foreground="white", alpha=0.8), pe.Normal()])

        # bold arrow
        acc_arrow = FancyArrowPatch((xs[0], ys[0]), (xs[0], ys[0]),
                                    arrowstyle="-|>", mutation_scale=26, linewidth=3.2,
                                    color="tab:red", alpha=0.95, zorder=5)
        acc_arrow.set_path_effects([pe.Stroke(linewidth=5, foreground="white", alpha=0.9), pe.Normal()])
        ax_scene.add_patch(acc_arrow)

        # right: cosines
        ax_plot.set_title("cos(angle to ref)")
        ax_plot.set_xlabel("frame"); ax_plot.set_ylabel("cos θ"); ax_plot.set_ylim(-1.05, 1.05); ax_plot.grid(True, alpha=0.3)
        ax_plot.plot(ts, cos_total, "o-", ms=3, lw=1.5, label="total", color="tab:blue")
        if show_components and _has_components(data):
            if cos_des is not None and np.isfinite(cos_des).any():
                ax_plot.plot(ts, cos_des, "--", lw=1.2, label="desired", color="tab:orange")
            if cos_ra is not None and np.isfinite(cos_ra).any():
                ax_plot.plot(ts, cos_ra, "--", lw=1.2, label="rep_agents", color="tab:green")
            if cos_ro is not None and np.isfinite(cos_ro).any():
                ax_plot.plot(ts, cos_ro, "--", lw=1.2, label="rep_obs", color="tab:red")

        if 0 <= x_reach_idx < len(ts):
            ax_plot.axvline(ts[x_reach_idx], color="k", lw=1.5, ls="--", alpha=0.7)
            ax_scene.plot([xs[x_reach_idx]], [ys[x_reach_idx]], "s", ms=8, mfc="none", mec="k", mew=1.5, zorder=3)

        (cursor_line,) = ax_plot.plot([ts[0], ts[0]], [-1.05, 1.05], color="k", lw=1.2)
        if ax_plot.get_legend_handles_labels()[0]:
            ax_plot.legend(loc="best")

        # animation frames
        def _update(i):
            # crowd
            pts = all_pts_per_frame[i]
            all_agents_scatter.set_offsets(pts if pts.size else np.zeros((0, 2)))
            # focus
            cur_point.set_data([xs[i]], [ys[i]])
            axv, ayv = ax_all[i], ay_all[i]
            tip_x, tip_y = xs[i] + viz_scale*axv, ys[i] + viz_scale*ayv
            acc_arrow.set_positions((xs[i], ys[i]), (tip_x, tip_y))
            acc_arrow.set_visible(bool(axv or ayv))
            # cursor
            cursor_line.set_data([ts[i], ts[i]], [-1.05, 1.05])
            return ()

        anim = FuncAnimation(fig, _update, frames=len(ts), interval=1000//fps, blit=False)
        writer = FFMpegWriter(fps=fps, bitrate=2200)
        anim.save(save_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"[AngleCosineAnimator] Saved video to: {save_path}")

    # ---------- IMAGE GRID (multi-model comparison) ----------
    @staticmethod
    def plot_model_comparison(
        models: Dict[str, Scene],
        person_id: int,
        ref_point: Point2D,
        save_path: str,
        dpi: int = 140,
        show_components: bool = True,
        arrow_scale: float = 1.2,   # only affects arrow snapshots (we draw trajectory only here)
    ) -> None:
        """
        Compare multiple models for the SAME person_id.
        Grid layout:
            left column  : trajectory overlay (all models)
            middle/right : for each force type (rows), cosine and magnitude vs time (overlaid models)
        Force types shown: total always; desired/rep_agents/rep_obs only if present in ANY model.
        """
        names = list(models.keys())
        colors = _model_colors(names)

        # compute alignment data for each model
        series: Dict[str, dict] = {}
        has_des = has_ra = has_ro = False
        # also grab common scene bounds & obstacles (first scene)
        any_scene: Scene = next(iter(models.values()))
        xmin, xmax, ymin, ymax = _scene_bounds(any_scene)
        span = max(xmax - xmin, ymax - ymin)

        for name, scene in models.items():
            d = compute_force_alignment(scene, ref_point, person_ids=[person_id]).get(person_id)
            if d is None:
                raise ValueError(f"[{name}] Person {person_id} not found.")
            series[name] = d
            has_des  = has_des or (d.get("cos_desired") is not None and np.isfinite(d["cos_desired"]).any())
            has_ra   = has_ra  or (d.get("cos_rep_agents") is not None and np.isfinite(d["cos_rep_agents"]).any())
            has_ro   = has_ro  or (d.get("cos_rep_obs") is not None and np.isfinite(d["cos_rep_obs"]).any())

        # rows = force types present
        rows: List[Tuple[str, str]] = [("total", "Total")]
        if show_components and has_des: rows.append(("desired", "Desired"))
        if show_components and has_ra:  rows.append(("rep_agents", "Repulsive (agents)"))
        if show_components and has_ro:  rows.append(("rep_obs", "Repulsive (obstacles)"))

        nrows = len(rows)
        ncols = 3  # [trajectory overlay | cosine | magnitude]
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.4*nrows), dpi=dpi,
                                 gridspec_kw=dict(wspace=0.25, hspace=0.35))
        if nrows == 1:
            axes = np.expand_dims(axes, 0)  # normalize to [row, col]

        # --- left column: trajectory overlay (top row only; share x/y for all rows)
        ax_traj = axes[0, 0]
        ax_traj.set_aspect("equal")
        ax_traj.set_title(f"Trajectories — pid {person_id}")
        ax_traj.set_xlim(xmin, xmax); ax_traj.set_ylim(ymin, ymax)
        ax_traj.grid(True, alpha=0.2)
        ax_traj.axvline(ref_point.x, color="k", lw=1.2, ls="--", alpha=0.7)

        # draw obstacles from any scene (first frame)
        first_frame = next(iter(any_scene.frames.values()))
        for o in first_frame.obstacles:
            ax_traj.plot([o.p1.x, o.p2.x], [o.p1.y, o.p2.y], color="0.5", lw=2, alpha=0.6)

        # all models overlay trajectories for this pid
        for name, scene in models.items():
            xs, ys = [], []
            # iterate frames sorted
            for t, fr in sorted(scene.frames.items()):
                p = fr.persons.get(person_id)
                if p is not None:
                    xs.append(p.position.x); ys.append(p.position.y)
            if xs:
                ax_traj.plot(xs, ys, "-", lw=2.0, alpha=0.9, color=colors[name], label=name)
                ax_traj.plot(xs[0], ys[0], "o", ms=5, color=colors[name], alpha=0.9)  # start marker
        if ax_traj.get_legend_handles_labels()[0]:
            ax_traj.legend(loc="best", fontsize=9)

        # --- middle & right columns per force row
        for r, (key, title) in enumerate(rows):
            ax_cos = axes[r, 1]
            ax_mag = axes[r, 2]
            ax_cos.set_title(f"{title}: cos to ref")
            ax_mag.set_title(f"{title}: |force|")

            ax_cos.set_xlabel("frame"); ax_cos.set_ylabel("cos θ")
            ax_mag.set_xlabel("frame"); ax_mag.set_ylabel("|a|")

            ax_cos.set_ylim(-1.05, 1.05); ax_cos.grid(True, alpha=0.3)
            ax_mag.grid(True, alpha=0.3)

            # plot per model
            for name, d in series.items():
                t  = d["t"]
                # cosine series
                if key == "total":
                    ycos = d["cos_total"]; ymag = d.get("amag_total", d.get("amag"))  # backward compat
                elif key == "desired":
                    ycos = d.get("cos_desired"); ymag = d.get("amag_desired")
                elif key == "rep_agents":
                    ycos = d.get("cos_rep_agents"); ymag = d.get("amag_rep_agents")
                elif key == "rep_obs":
                    ycos = d.get("cos_rep_obs"); ymag = d.get("amag_rep_obs")
                else:
                    continue

                if ycos is not None and np.isfinite(ycos).any():
                    ax_cos.plot(t, ycos, "-", lw=1.6, color=colors[name], alpha=0.95, label=name)
                if ymag is not None and np.isfinite(ymag).any():
                    ax_mag.plot(t, ymag, "-", lw=1.6, color=colors[name], alpha=0.95, label=name)

                # mark x≈Px crossing if present
                xr = d["x_reach_idx"]
                if xr.size:
                    ax_cos.axvline(int(t[int(xr[0])]), color=colors[name], lw=1.0, ls="--", alpha=0.6)
                    ax_mag.axvline(int(t[int(xr[0])]), color=colors[name], lw=1.0, ls="--", alpha=0.6)

            # dedup legends
            for ax in (ax_cos, ax_mag):
                h, l = ax.get_legend_handles_labels()
                if h:
                    # show legend only on the first row or only in cos panel to reduce clutter
                    if r == 0 and ax is ax_cos:
                        ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[AngleCosineAnimator] Saved comparison image to: {save_path}")
