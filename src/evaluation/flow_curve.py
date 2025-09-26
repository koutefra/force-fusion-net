import numpy as np

class FlowCurveEvaluator:
    def __init__(self, rect: tuple[float, float, float, float], axis: tuple[float, float]):
        """
        rect: (xmin, xmax, ymin, ymax) measurement area in meters
        axis: corridor direction (will be normalized)
        """
        self.rect = rect
        ax = np.array(axis, float)
        self.axis = ax / (np.linalg.norm(ax) + 1e-12)
        self.area = (rect[1] - rect[0]) * (rect[3] - rect[2])
        self.width = max(rect[1] - rect[0], rect[3] - rect[2])

    def evaluate_flow_curve(self, scene: "Scene", bin_width: float = 0.4) -> dict[str, float]:
        """Compute flow curve statistics (capacity, crit. density, free speed)."""
        rho_t, vpar_t, Js_t = [], [], []

        for frame in scene.frames.values():
            inside = [
                p for p in frame.persons.values()
                if self.rect[0] <= p.position.x <= self.rect[1]
                and self.rect[2] <= p.position.y <= self.rect[3]
            ]
            N = len(inside)
            rho = N / self.area if self.area > 0 else 0.0

            if N > 0:
                vpars = [np.dot([p.velocity.x, p.velocity.y], self.axis) for p in inside if p.velocity]
                vmean = np.mean(vpars) if vpars else 0.0
            else:
                vmean = 0.0

            Js = rho * vmean
            rho_t.append(rho); vpar_t.append(vmean); Js_t.append(Js)

        rho_t, vpar_t, Js_t = np.array(rho_t), np.array(vpar_t), np.array(Js_t)

        # Bin by density
        rho_max = max(3.0, rho_t.max() + bin_width)
        bins = np.arange(0, rho_max, bin_width)
        mids = (bins[:-1] + bins[1:]) / 2
        Js_bin, v_bin = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            sel = (rho_t >= b0) & (rho_t < b1)
            Js_bin.append(np.mean(Js_t[sel]) if np.any(sel) else np.nan)
            v_bin.append(np.mean(vpar_t[sel]) if np.any(sel) else np.nan)
        Js_bin, v_bin = np.array(Js_bin), np.array(v_bin)

        # Capacity and critical density
        if np.all(np.isnan(Js_bin)):
            capacity_Js, rho_crit = 0.0, 0.0
        else:
            idx = np.nanargmax(Js_bin)
            capacity_Js = float(Js_bin[idx])
            rho_crit = float(mids[idx])

        return {
            "capacity_Js": capacity_Js,
            "rho_crit": rho_crit,
            "rho_bins": mids.tolist(),
            "Js_bins": Js_bin.tolist(),
            "v_bins": v_bin.tolist(),
        }
