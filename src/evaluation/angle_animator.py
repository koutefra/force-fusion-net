# evaluation/angle_animator.py
from __future__ import annotations
import os
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch

from entities.scene import Scene
from entities.vector2d import Point2D
from evaluation.align import compute_force_alignment


class AngleCosineAnimator:
    def __init__(self, scene: Scene, ref_point: Point2D = Point2D(0.0, 0.0)):
        self.scene = scene
        self.ref = ref_point
        self.frames = dict(sorted(scene.frames.items()))
        self.fnums = np.array(list(self.frames.keys()), dtype=int)

    def _scene_bounds(self):
        bl, tr = self.scene.bounding_box
        pad = 0.02 * max(tr.x - bl.x, tr.y - bl.y)
        return (bl.x - pad, tr.x + pad, bl.y - pad, tr.y + pad)

    def animate_person(
        self,
        person_id: int,
        save_path: Optional[str] = None,
        show_components: bool = True,
        dpi: int = 110,
        arrow_scale: float = 1.0,  # bump this if you want even bigger arrows
    ):
        data = compute_force_alignment(self.scene, self.ref, person_ids=[person_id]).get(person_id)
        if data is None:
            raise ValueError(f"Person {person_id} not found in scene.")

        ts = data["t"]
        cos_total = data["cos_total"]
        cos_des   = data["cos_desired"]
        cos_ra    = data["cos_rep_agents"]
        cos_ro    = data["cos_rep_obs"]
        x_reach_idx = int(data["x_reach_idx"][0]) if data["x_reach_idx"].size else -1

        # --- Precompute focused person's trajectory & total acceleration over those frames
        xs, ys, ax_all, ay_all = [], [], [], []
        for t in ts:
            p = self.frames[t].persons.get(person_id)
            xs.append(p.position.x)
            ys.append(p.position.y)
            if p.acceleration is not None:
                ax_all.append(p.acceleration.x)
                ay_all.append(p.acceleration.y)
            else:
                ax_all.append(0.0)
                ay_all.append(0.0)
        xs, ys = np.asarray(xs), np.asarray(ys)
        ax_all, ay_all = np.asarray(ax_all), np.asarray(ay_all)

        # --- Also precompute ALL agents' positions per frame (for faint context)
        # arrays-of-arrays: list of Nx2 points per frame in ts
        all_pts_per_frame = []
        for t in ts:
            fr = self.frames[t]
            if fr.persons:
                pts = np.array([[p.position.x, p.position.y] for p in fr.persons.values()], dtype=float)
            else:
                pts = np.zeros((0, 2), dtype=float)
            all_pts_per_frame.append(pts)

        # --- Robust auto-scale for arrow length
        acc_norms = np.hypot(ax_all, ay_all)
        acc_ref = np.nanpercentile(acc_norms[acc_norms > 0], 90) if np.any(acc_norms > 0) else 1.0
        xmin, xmax, ymin, ymax = self._scene_bounds()
        # bigger default arrow: ~12% of scene span for a p90 accel, times arrow_scale
        target_len = 0.12 * max(xmax - xmin, ymax - ymin)
        viz_scale = (target_len / max(acc_ref, 1e-9)) * arrow_scale

        # --- Figure / axes
        fig = plt.figure(figsize=(10, 5), dpi=dpi)
        gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], width_ratios=[1, 1], hspace=0.1, wspace=0.2)
        ax_scene = fig.add_subplot(gs[0, 0])
        ax_plot  = fig.add_subplot(gs[0, 1])
        ax_slider= fig.add_subplot(gs[1, :])

        plt.subplots_adjust(bottom=0.2)  # room for slider

        # --- Left: scene
        ax_scene.set_xlim(xmin, xmax); ax_scene.set_ylim(ymin, ymax)
        ax_scene.set_aspect("equal", adjustable="box")
        ax_scene.set_title(f"Scene {self.scene.id} — Person {person_id}")
        ax_scene.grid(True, alpha=0.2)

        # obstacles (once)
        for fr in self.frames.values():
            for o in fr.obstacles:
                ax_scene.plot([o.p1.x, o.p2.x], [o.p1.y, o.p2.y], color="0.4", lw=2, alpha=0.6)
            break

        # show where ref.x lies
        ax_scene.axvline(self.ref.x, color="k", lw=1.5, ls="--", alpha=0.7)

        # all agents (tiny, faint)
        all_agents_scatter = ax_scene.scatter([], [], s=8, c="tab:cyan", alpha=0.35, zorder=1)

        # focused person's full traj (faint) + current point (emphasized)
        (traj_line,) = ax_scene.plot(xs, ys, "-", lw=1.5, alpha=0.35, color="tab:blue", zorder=2)
        (cur_point,) = ax_scene.plot([xs[0]], [ys[0]], "o", ms=8, color="tab:red", zorder=4,
                                     path_effects=[pe.Stroke(linewidth=3, foreground="white", alpha=0.8), pe.Normal()])

        # thicker, haloed arrow for total acceleration
        acc_arrow = FancyArrowPatch(
            (xs[0], ys[0]), (xs[0], ys[0]),
            arrowstyle="-|>", mutation_scale=24, linewidth=3.0, color="tab:red", alpha=0.95, zorder=5
        )
        acc_arrow.set_path_effects([pe.Stroke(linewidth=5, foreground="white", alpha=0.9), pe.Normal()])
        ax_scene.add_patch(acc_arrow)

        # --- Right: cos vs time
        ax_plot.set_title("cos(angle(person→P, force))")
        ax_plot.set_xlabel("frame"); ax_plot.set_ylabel("cos θ")
        ax_plot.set_ylim(-1.05, 1.05); ax_plot.grid(True, alpha=0.3)

        (l_total,) = ax_plot.plot(ts, cos_total, "o-", ms=3, lw=1.5, label="total", color="tab:blue")
        if show_components:
            if np.isfinite(cos_des).any():
                ax_plot.plot(ts, cos_des, "--", lw=1.2, label="desired", color="tab:orange")
            if np.isfinite(cos_ra).any():
                ax_plot.plot(ts, cos_ra, "--", lw=1.2, label="rep_agents", color="tab:green")
            if np.isfinite(cos_ro).any():
                ax_plot.plot(ts, cos_ro, "--", lw=1.2, label="rep_obs", color="tab:red")

        if 0 <= x_reach_idx < len(ts):
            ax_plot.axvline(ts[x_reach_idx], color="k", lw=1.5, ls="--", alpha=0.7)
            ax_scene.plot([xs[x_reach_idx]], [ys[x_reach_idx]], "s",
                          ms=8, mfc="none", mec="k", mew=1.5, zorder=3)

        (cursor_line,) = ax_plot.plot([ts[0], ts[0]], [-1.05, 1.05], color="k", lw=1.2)

        # legend if any labeled lines exist
        handles, labels = ax_plot.get_legend_handles_labels()
        if handles: ax_plot.legend(loc="best")

        # --- Slider
        slider = Slider(ax_slider, "frame", valmin=int(ts[0]), valmax=int(ts[-1]), valinit=int(ts[0]), valstep=1)

        def _update(frame_num: int):
            # index into our time series
            idx = int(np.clip(np.searchsorted(ts, frame_num), 0, len(ts) - 1))

            # all agents at this frame (faint context)
            pts = all_pts_per_frame[idx]
            if pts.size:
                all_agents_scatter.set_offsets(pts)
            else:
                all_agents_scatter.set_offsets(np.zeros((0, 2)))

            # focused marker
            cur_point.set_data([xs[idx]], [ys[idx]])

            # arrow (no label, just bold arrow)
            axv, ayv = ax_all[idx], ay_all[idx]
            tip_x = xs[idx] + viz_scale * axv
            tip_y = ys[idx] + viz_scale * ayv
            acc_arrow.set_positions((xs[idx], ys[idx]), (tip_x, tip_y))
            acc_arrow.set_visible(bool(axv or ayv))

            # right panel cursor
            cursor_line.set_data([frame_num, frame_num], [-1.05, 1.05])

            fig.canvas.draw_idle()

        slider.on_changed(lambda v: _update(int(v)))
        _update(int(ts[0]))

        # interactive if possible; otherwise save
        backend = matplotlib.get_backend().lower()
        headless = ("display" not in os.environ) or any(k in backend for k in ["agg", "pdf", "svg", "ps"])
        if save_path or headless:
            path = save_path or f"angle_anim_{self.scene.id}_pid{person_id}.mp4"
            try:
                anim = FuncAnimation(fig, lambda i: _update(int(ts[i])), frames=len(ts), interval=30, blit=False)
                writer = FFMpegWriter(fps=10, bitrate=1800)
                anim.save(path, writer=writer, dpi=dpi)
                print(f"[angle_animator] Saved MP4 to: {path}")
            except Exception as e:
                print(f"[angle_animator] ffmpeg failed ({e}); falling back to GIF.")
                anim = FuncAnimation(fig, lambda i: _update(int(ts[i])), frames=len(ts), interval=30, blit=False)
                anim.save(path.replace(".mp4", ".gif"), dpi=dpi)
            plt.close(fig)
        else:
            plt.show(block=True)
