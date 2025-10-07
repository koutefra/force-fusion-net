from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Tuple
from entities.scene import Scene
from entities.vector2d import Point2D, Acceleration

def _safe_unit(vx: float, vy: float, eps: float = 1e-12) -> Tuple[float, float]:
    n = np.hypot(vx, vy)
    if n < eps: return 0.0, 0.0
    return vx / n, vy / n

def _angle_and_cos(ax: float, ay: float, bx: float, by: float, eps: float = 1e-12) -> Tuple[float, float]:
    ax, ay = _safe_unit(ax, ay, eps)
    bx, by = _safe_unit(bx, by, eps)
    c = np.clip(ax * bx + ay * by, -1.0, 1.0)
    # principal angle (0..π)
    ang = float(np.arccos(c))
    return ang, float(c)

def compute_force_alignment(
    scene: Scene,
    ref_point: Point2D = Point2D(0.0, 0.0),
    person_ids: Optional[List[int]] = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Returns per-person dict with:
      t              : frame numbers (np.ndarray[int])
      cos_total      : cosine with total acceleration (NaN if missing)
      ang_total      : angle [rad] with total acceleration (NaN if missing)
      cos_desired    : ... (if forces present, else NaN)
      cos_rep_agents : ...
      cos_rep_obs    : ...
      x_reach_idx    : index where |x - ref_point.x| is minimal (int), useful to mark 'reached x(P)'
    """
    persons = person_ids or sorted(scene.get_all_person_ids())
    frames_sorted = sorted(scene.frames.keys())

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for pid in persons:
        ts, cos_tot, ang_tot = [], [], []
        cos_des, cos_ra, cos_ro = [], [], []
        xs = []

        for t in frames_sorted:
            frame = scene.frames[t]
            p = frame.persons.get(pid)
            if p is None:
                # keep time axis consistent (optional: skip)
                continue

            # direction from person -> reference point
            dx, dy = (ref_point.x - p.position.x), (ref_point.y - p.position.y)
            ux, uy = _safe_unit(dx, dy)

            # total acceleration
            if p.acceleration is not None:
                ang, c = _angle_and_cos(ux, uy, p.acceleration.x, p.acceleration.y)
            else:
                ang, c = np.nan, np.nan

            # per-component, if available
            if getattr(p, "forces", None) is not None:
                d = p.forces.desired
                ra = p.forces.repulsive_agents
                ro = p.forces.repulsive_obs

                def comp_cos(comp):
                    if comp is None: return np.nan
                    _, cc = _angle_and_cos(ux, uy, comp.x, comp.y)
                    return cc

                c_des = comp_cos(d)
                c_ra  = comp_cos(ra)
                c_ro  = comp_cos(ro)
            else:
                c_des = c_ra = c_ro = np.nan

            ts.append(t)
            cos_tot.append(c)
            ang_tot.append(ang)
            cos_des.append(c_des)
            cos_ra.append(c_ra)
            cos_ro.append(c_ro)
            xs.append(p.position.x)

        if not ts:
            continue

        ts = np.asarray(ts, dtype=int)
        cos_tot = np.asarray(cos_tot, dtype=float)
        ang_tot = np.asarray(ang_tot, dtype=float)
        cos_des = np.asarray(cos_des, dtype=float)
        cos_ra  = np.asarray(cos_ra, dtype=float)
        cos_ro  = np.asarray(cos_ro, dtype=float)
        xs = np.asarray(xs, dtype=float)

        # index where person x is closest to ref x
        x_reach_idx = int(np.argmin(np.abs(xs - ref_point.x))) if xs.size else -1

        out[pid] = dict(
            t=ts,
            cos_total=cos_tot,
            ang_total=ang_tot,
            cos_desired=cos_des,
            cos_rep_agents=cos_ra,
            cos_rep_obs=cos_ro,
            x_reach_idx=np.array([x_reach_idx], dtype=int),
        )
    return out
